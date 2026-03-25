"""
Модуль с тестовыми задачами из теории управления.
Каждая задача формализована как функция для оптимизации.
"""

import numpy as np
from scipy.integrate import odeint
import control as ctrl
import warnings
from simulation import simulate_dc_motor_pid, compute_step_metrics

warnings.filterwarnings('ignore')

# ============================================================================
# 1. ОПТИМИЗАЦИЯ ПИД-РЕГУЛЯТОРА ДЛЯ ДВИГАТЕЛЯ ПОСТОЯННОГО ТОКА
# ============================================================================

def dc_motor_pid_objective(params):
    """
    Целевая функция для оптимизации ПИД-регулятора двигателя постоянного тока.
    Минимизируется ITAE + штраф за перерегулирование + штраф за ошибку на хвосте.
    
    Математическая постановка:
        θ = [Kp, Ki, Kd]^T
        G(s) = Kt / (JLs^2 + (JR+BL)s + (BR+Kt*Kb))
        C(s) = (Kd s^2 + Kp s + Ki)/s
        T(s) = C(s)G(s) / (1 + C(s)G(s))
        J(θ) = Δt Σ t_k |e_k| + 100·max(0, max(y)-1) + 50·(1/|I|) Σ_{k∈I} |e_k|,
            где e_k = 1 - y_k, I = {k : t_k > 3}
    """
    Kp, Ki, Kd = params
    
    # Проверка границ (соответствует bounds из get_problem_info)
    if not (0.1 <= Kp <= 50 and 0.01 <= Ki <= 30 and 0 <= Kd <= 10):
        return 1e6
    
    try:
        t, y = simulate_dc_motor_pid(params, t_end=5, n_points=500)
        e = 1 - y
        dt = t[1] - t[0]
        
        # ITAE
        J = np.sum(t * np.abs(e)) * dt
        
        # Штраф за перерегулирование
        overshoot = max(0, (np.max(y) - y[-1]) / y[-1] * 100) if y[-1] > 0 else 100
        J += 100 * overshoot
        
        # Штраф за среднюю ошибку на хвосте (t > 3)
        tail_mask = t > 3
        if np.any(tail_mask):
            tail_error = np.mean(np.abs(e[tail_mask]))
            J += 50 * tail_error
        
        return float(J)
        
    except Exception:
        return 1e6


# ============================================================================
# 2. БАЛАНСИРОВКА ПЕРЕВЕРНУТОГО МАЯТНИКА
# ============================================================================

def inverted_pendulum_objective(params):
    """
    Целевая функция для балансировки перевернутого маятника.
    Оптимизируются коэффициенты обратной связи по состоянию K = [K1, K2, K3, K4].
    Используется линеаризованная модель:
        dx/dt = (A - B K) x, x(0) = [0,0,0.1,0]^T
        J(K) = (1/N) Σ x_k^T Q x_k, Q = diag(10, 0.1, 100, 0.1)
    Если maxRe(λ(A-BK)) ≥ 0, возвращается штраф 1e6.
    """
    K1, K2, K3, K4 = params
    
    # Проверка границ (соответствует bounds из get_problem_info)
    if any(np.abs(p) > 50 for p in params):
        return 1e6
    
    try:
        M = 1.0      # Масса тележки (кг)
        m = 0.1      # Масса стержня (кг)
        b = 0.1      # Трение тележки (Н/м/с)
        l = 0.5      # Длина до центра массы стержня (м)
        g = 9.81     # Ускорение свободного падения (м/с²)
        
        # Матрицы линеаризованной модели
        A = np.array([
            [0, 1, 0, 0],
            [0, -b/M, -m*g/M, 0],
            [0, 0, 0, 1],
            [0, -b/(M*l), (M+m)*g/(M*l), 0]
        ])
        
        B = np.array([[0], [1/M], [0], [1/(M*l)]])
        
        K = np.array([[K1, K2, K3, K4]])
        A_closed = A - B @ K
        
        # Проверка устойчивости
        eigenvalues = np.linalg.eigvals(A_closed)
        if np.any(np.real(eigenvalues) >= 0):
            return 1e6
        
        # Моделирование
        def system_dynamics(x, t):
            return A_closed.dot(x)
        
        x0 = np.array([0, 0, 0.1, 0])
        t = np.linspace(0, 5, 500)
        x = odeint(system_dynamics, x0, t)
        
        Q = np.diag([10, 0.1, 100, 0.1])
        cost = np.mean([x[i].T @ Q @ x[i] for i in range(len(t))])
        
        return float(cost)
        
    except Exception:
        return 1e6


# ============================================================================
# 3. УПРАВЛЕНИЕ УРОВНЕМ ЖИДКОСТИ В РЕЗЕРВУАРАХ (усложненная модель)
# ============================================================================

def liquid_level_control_objective(params):
    """
    Целевая функция для управления уровнем жидкости в двух связанных резервуарах.
    Оптимизируются параметры двух PI-регуляторов: [Kp1, Ki1, Kp2, Ki2].
    Модель с нелинейными потоками (квадратичная зависимость).
    """
    Kp1, Ki1, Kp2, Ki2 = params
    
    # Проверка границ (соответствует bounds из get_problem_info)
    if not (0 <= Kp1 <= 5 and 0 <= Ki1 <= 2 and 0 <= Kp2 <= 5 and 0 <= Ki2 <= 2):
        return 1e6
    
    try:
        # Параметры системы
        A1, A2 = 2.0, 1.5          # площади сечений, м²
        R1, R2 = 0.5, 0.7          # гидравлические сопротивления
        h1_ref, h2_ref = 1.0, 0.8  # желаемые уровни
        dt = 0.1
        steps = 100
        
        # Начальные условия
        h1, h2 = 0.5, 0.3
        I1, I2 = 0.0, 0.0
        total_error = 0.0
        
        for _ in range(steps):
            e1 = h1_ref - h1
            e2 = h2_ref - h2
            
            I1 += e1 * dt
            I2 += e2 * dt
            
            u1 = np.clip(Kp1 * e1 + Ki1 * I1, 0, 2)
            u2 = np.clip(Kp2 * e2 + Ki2 * I2, 0, 2)
            
            # Нелинейные потоки (квадратичные)
            q12 = max(0, (h1 - h2) / R1) * np.sqrt(abs(h1 - h2) + 1e-6)
            q2out = (h2 / R2) * np.sqrt(h2 + 1e-6)
            
            h1 += (u1 - q12) / A1 * dt
            h2 += (q12 - q2out + u2) / A2 * dt
            
            # Гарантия неотрицательности
            h1 = max(0, h1)
            h2 = max(0, h2)
            
            total_error += (abs(e1) + abs(e2)) * dt
        
        # Штраф за установившуюся ошибку
        total_error += 10 * (abs(h1_ref - h1) + abs(h2_ref - h2))
        
        return float(total_error)
        
    except Exception:
        return 1e6


# ============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================================

def get_problem_info(problem_name):
    """
    Возвращает информацию о задаче: размерность, границы параметров.
    """
    problems = {
        'dc_motor_pid': {
            'dim': 3,
            'bounds': (np.array([0.1, 0.01, 0]), np.array([50, 30, 10])),
            'description': 'Оптимизация ПИД-регулятора для двигателя постоянного тока',
            'objective_func': dc_motor_pid_objective
        },
        'inverted_pendulum': {
            'dim': 4,
            'bounds': (np.array([-50, -50, -50, -50]), np.array([50, 50, 50, 50])),
            'description': 'Балансировка перевернутого маятника',
            'objective_func': inverted_pendulum_objective
        },
        'liquid_level': {
            'dim': 4,
            'bounds': (np.array([0, 0, 0, 0]), np.array([5, 2, 5, 2])),
            'description': 'Управление уровнем жидкости в резервуарах',
            'objective_func': liquid_level_control_objective
        }
    }
    
    return problems.get(problem_name)