"""
Модуль с тестовыми задачами из теории управления.
Каждая задача формализована как функция для оптимизации.
"""

import numpy as np
from scipy.integrate import odeint
import control as ctrl
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. ОПТИМИЗАЦИЯ ПИД-РЕГУЛЯТОРА ДЛЯ ДВИГАТЕЛЯ ПОСТОЯННОГО ТОКА
# ============================================================================

def dc_motor_pid_objective(params):
    """
    Целевая функция для оптимизации ПИД-регулятора двигателя постоянного тока.
    Минимизация ITAE (Integral of Time-weighted Absolute Error).
    """
    Kp, Ki, Kd = params
    
    # Ограничение параметров (более мягкие границы)
    if Kp < 0.1 or Ki < 0.01 or Kd < 0 or Kp > 50 or Ki > 30 or Kd > 10:
        return 1e6  # Уменьшен штраф
    
    try:
        # Параметры двигателя постоянного тока (упрощенная модель)
        R = 1.0      # Сопротивление (Ом)
        L = 0.5      # Индуктивность (Гн)
        Kb = 0.01    # Коэффициент противо-ЭДС (В/рад/с)
        Kt = 0.01    # Коэффициент момента (Нм/А)
        J = 0.01     # Момент инерции (кг·м²)
        B = 0.1      # Коэффициент вязкого трения (Нм·с)
        
        # Передаточная функция двигателя
        num = [Kt]
        den = [J*L, J*R + B*L, B*R + Kt*Kb]
        motor_tf = ctrl.TransferFunction(num, den)
        
        # ПИД-регулятор
        pid_tf = ctrl.TransferFunction([Kd, Kp, Ki], [1, 1e-10])
        
        # Система с обратной связью
        sys_open = ctrl.series(pid_tf, motor_tf)
        sys_closed = ctrl.feedback(sys_open, 1)
        
        # Временной интервал
        t = np.linspace(0, 5, 500)  # Уменьшено время моделирования
        
        # Ступенчатый отклик
        t_out, y_out = ctrl.step_response(sys_closed, t)
        
        # Вычисление ITAE
        error = 1 - y_out
        # Используем численное интегрирование через sum
        itae = np.sum(t_out * np.abs(error)) * (t_out[1] - t_out[0])
        
        # Штраф за перерегулирование
        overshoot = max(0, np.max(y_out) - 1)
        itae += 100 * overshoot
        
        # Штраф за колебания
        settling_idx = np.where(t_out > 3)[0]
        if len(settling_idx) > 0:
            settling_error = np.mean(np.abs(error[settling_idx]))
            itae += 50 * settling_error
        
        return float(itae)
        
    except Exception as e:
        # Возвращаем штраф за неустойчивость
        return 1e6


# ============================================================================
# 2. БАЛАНСИРОВКА ПЕРЕВЕРНУТОГО МАЯТНИКА
# ============================================================================

def inverted_pendulum_objective(params):
    """
    Целевая функция для балансировки перевернутого маятника.
    """
    K1, K2, K3, K4 = params
    
    # Ограничение параметров
    if any(np.abs(p) > 100 for p in params):
        return 1e6
    
    try:
        # Параметры маятника (упрощенная модель)
        M = 1.0      # Масса тележки (кг)
        m = 0.1      # Масса стержня (кг)
        b = 0.1      # Трение тележки (Н/м/с)
        l = 0.5      # Длина до центра массы стержня (м)
        g = 9.81     # Ускорение свободного падения (м/с²)
        
        # Матрицы системы (линеаризованная модель)
        A = np.array([
            [0, 1, 0, 0],
            [0, -b/M, -m*g/M, 0],
            [0, 0, 0, 1],
            [0, -b/(M*l), (M+m)*g/(M*l), 0]
        ])
        
        B = np.array([[0], [1/M], [0], [1/(M*l)]])
        
        # Коэффициенты регулятора
        K = np.array([[K1, K2, K3, K4]])
        
        # Система с регулятором
        A_closed = A - B @ K
        
        # Проверка устойчивости
        eigenvalues = np.linalg.eigvals(A_closed)
        if np.any(np.real(eigenvalues) >= 0):
            return 1e6  # Штраф за неустойчивость
        
        # Моделирование
        def system_dynamics(x, t):
            return A_closed.dot(x)
        
        # Начальные условия
        x0 = np.array([0, 0, 0.1, 0])  # Угол 0.1 рад
        
        # Временной интервал
        t = np.linspace(0, 5, 500)
        
        # Решение системы
        x = odeint(system_dynamics, x0, t)
        
        # Целевая функция
        Q = np.diag([10, 0.1, 100, 0.1])
        
        # Интеграл квадратов состояний (упрощенное вычисление)
        cost = 0
        for i in range(len(t)):
            cost += x[i].T @ Q @ x[i]
        
        cost = cost / len(t)
        
        return float(cost)
        
    except Exception as e:
        return 1e6


# ============================================================================
# 3. УПРАВЛЕНИЕ УРОВНЕМ ЖИДКОСТИ В РЕЗЕРВУАРАХ
# ============================================================================

def liquid_level_control_objective(params):
    """
    Целевая функция для управления уровнем жидкости в двух связанных резервуарах.
    """
    Kp1, Ki1, Kp2, Ki2 = params
    
    # Ограничение параметров
    if any(p < 0 or p > 10 for p in params):
        return 1e6
    
    try:
        # Параметры системы (упрощенная модель)
        A1 = 2.0     # Площадь первого резервуара (м²)
        A2 = 1.5     # Площадь второго резервуара (м²)
        R1 = 0.5     # Сопротивление первой трубы (с/м²)
        R2 = 0.7     # Сопротивление второй трубы (с/м²)
        
        # Желаемые уровни
        h1_desired = 1.0
        h2_desired = 0.8
        
        # Моделирование системы (дискретное)
        dt = 0.1
        steps = 100
        
        # Начальные условия
        h1, h2 = 0.5, 0.3
        error_integral1 = 0
        error_integral2 = 0
        
        total_error = 0
        
        for step in range(steps):
            # Ошибки
            e1 = h1_desired - h1
            e2 = h2_desired - h2
            
            # Интегралы ошибок
            error_integral1 += e1 * dt
            error_integral2 += e2 * dt
            
            # Управляющие воздействия
            u1 = Kp1 * e1 + Ki1 * error_integral1
            u2 = Kp2 * e2 + Ki2 * error_integral2
            
            # Ограничение управляющих воздействий
            u1 = np.clip(u1, 0, 2)
            u2 = np.clip(u2, 0, 2)
            
            # Упрощенные уравнения баланса
            q12 = max(0, (h1 - h2) / R1)
            q2_out = h2 / R2
            
            # Обновление уровней
            h1 += (u1 - q12) / A1 * dt
            h2 += (q12 - q2_out + u2) / A2 * dt
            
            # Накопление ошибки
            total_error += (abs(e1) + abs(e2)) * dt
        
        # Добавляем штраф за установившуюся ошибку
        final_e1 = abs(h1_desired - h1)
        final_e2 = abs(h2_desired - h2)
        total_error += 10 * (final_e1 + final_e2)
        
        return float(total_error)
        
    except Exception as e:
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