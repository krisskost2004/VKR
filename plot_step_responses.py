"""
Скрипт для построения графиков переходных процессов двигателя постоянного тока
с оптимальными параметрами ПИД-регулятора, найденными каждым алгоритмом.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os
import control as ctrl
import pandas as pd

# Настройка стиля графиков
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'figure.figsize': (14, 8),
    'figure.dpi': 100,
    'savefig.dpi': 300
})

# Цвета для алгоритмов (как в visualization.py)
colors = {
    'PSO': '#FF6B6B',    # Красный
    'GWO': '#4ECDC4',    # Бирюзовый
    'WOA': '#45B7D1',    # Голубой
    'HHO': '#96CEB4',    # Зеленый
    'SMA': '#FFEAA7'     # Желтый
}

def load_best_solutions():
    """Загружает лучшие решения для задачи dc_motor_pid из результатов эксперимента"""
    
    results_file = "experiment_results/results.json"
    
    if not os.path.exists(results_file):
        print(f"❌ Файл {results_file} не найден. Сначала запустите experiment.py")
        return None
    
    with open(results_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Извлекаем данные для задачи с двигателем
    if 'dc_motor_pid' not in data:
        print("❌ Данные для задачи dc_motor_pid не найдены")
        return None
    
    motor_data = data['dc_motor_pid']
    best_solutions = {}
    
    print("\n📊 Загрузка лучших решений для каждого алгоритма:")
    print("-" * 60)
    
    for algo_name, algo_data in motor_data.items():
        if 'error' not in algo_data and 'all_runs' in algo_data:
            # Берем лучший запуск (с минимальным фитнесом)
            runs = algo_data['all_runs']
            best_run = min(runs, key=lambda x: x.get('best_fitness', float('inf')))
            
            if 'solution' in best_run:
                best_solutions[algo_name] = {
                    'params': best_run['solution'],
                    'fitness': best_run['best_fitness']
                }
                print(f"  {algo_name}: Kp={best_run['solution'][0]:.4f}, "
                      f"Ki={best_run['solution'][1]:.4f}, Kd={best_run['solution'][2]:.4f}, "
                      f"фитнес={best_run['best_fitness']:.6e}")
    
    print("-" * 60)
    return best_solutions

def get_motor_response(Kp, Ki, Kd, t):
    """
    Вычисляет переходную характеристику двигателя с заданными параметрами ПИД-регулятора.
    
    Args:
        Kp, Ki, Kd: параметры ПИД-регулятора
        t: временной массив
    
    Returns:
        t_out: временной массив
        y_out: выходной сигнал (скорость)
    """
    # Параметры двигателя (те же, что и в problems.py)
    R = 1.0      # Сопротивление (Ом)
    L = 0.5      # Индуктивность (Гн)
    Kb = 0.01    # Коэффициент противо-ЭДС (В/рад/с)
    Kt = 0.01    # Коэффициент момента (Нм/А)
    J = 0.01     # Момент инерции (кг·м²)
    B = 0.1      # Коэффициент вязкого трения (Нм·с)
    
    # Передаточная функция двигателя
    motor_num = [Kt]
    motor_den = [J*L, J*R + B*L, B*R + Kt*Kb]
    motor_tf = ctrl.TransferFunction(motor_num, motor_den)
    
    # ПИД-регулятор
    pid_num = [Kd, Kp, Ki]
    pid_den = [1, 1e-10]  # Малое число для избежания деления на ноль
    pid_tf = ctrl.TransferFunction(pid_num, pid_den)
    
    try:
        # Система с обратной связью
        sys_open = ctrl.series(pid_tf, motor_tf)
        sys_closed = ctrl.feedback(sys_open, 1)
        
        # Ступенчатый отклик
        t_out, y_out = ctrl.step_response(sys_closed, t)
        return t_out, y_out
    except Exception as e:
        print(f"⚠️ Ошибка при расчете отклика для Kp={Kp:.2f}, Ki={Ki:.2f}, Kd={Kd:.2f}: {e}")
        return t, np.zeros_like(t)

def calculate_step_metrics(t, y):
    """
    Вычисляет метрики переходного процесса (как в статье Nature)
    
    Returns:
        overshoot: перерегулирование (%)
        rise_time: время нарастания (с) (от 10% до 90%)
        settling_time: время установления (с) (до 2% от установившегося значения)
    """
    # Установившееся значение (последние 10% времени)
    steady_state = np.mean(y[int(0.9*len(y)):])
    
    if steady_state < 0.01:  # Если система не вышла на режим
        return 0, float('inf'), float('inf')
    
    # Перерегулирование
    max_val = np.max(y)
    overshoot = max(0, (max_val - steady_state) / steady_state * 100)
    
    # Время нарастания (10% -> 90%)
    threshold_10 = 0.1 * steady_state
    threshold_90 = 0.9 * steady_state
    
    idx_10 = np.where(y >= threshold_10)[0]
    idx_90 = np.where(y >= threshold_90)[0]
    
    if len(idx_10) > 0 and len(idx_90) > 0:
        rise_time = t[idx_90[0]] - t[idx_10[0]]
    else:
        rise_time = float('inf')
    
    # Время установления (вхождение в 2% коридор)
    settling_threshold = 0.02 * steady_state
    settled = np.abs(y - steady_state) < settling_threshold
    
    # Ищем последний момент, когда система была вне коридора
    unsettled_indices = np.where(~settled)[0]
    if len(unsettled_indices) > 0:
        settling_time = t[unsettled_indices[-1]]
    else:
        settling_time = 0
    
    return overshoot, rise_time, settling_time

def plot_step_responses(best_solutions):
    """Строит графики переходных процессов для всех алгоритмов"""
    
    if not best_solutions:
        print("❌ Нет данных для построения графиков")
        return None
    
    # Временной интервал
    t = np.linspace(0, 2, 1000)  # 2 секунды, 1000 точек
    
    # Создаем фигуру с двумя подграфиками
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Сравнение переходных процессов двигателя постоянного тока\nс оптимальными параметрами ПИД-регуляторов', 
                fontsize=16, fontweight='bold')
    
    # График 1: Полный переходной процесс
    ax1.set_title('Переходная характеристика (0-2 с)')
    ax1.set_xlabel('Время, с')
    ax1.set_ylabel('Угловая скорость (нормированная)')
    ax1.axhline(y=1.0, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Установившееся значение')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 2)
    ax1.set_ylim(0, 1.2)
    
    # График 2: Начальный участок (детали)
    ax2.set_title('Начальный участок (0-0.3 с)')
    ax2.set_xlabel('Время, с')
    ax2.set_ylabel('Угловая скорость (нормированная)')
    ax2.set_xlim(0, 0.3)
    ax2.set_ylim(0, 1.2)
    ax2.axhline(y=1.0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax2.grid(True, alpha=0.3)
    
    # Словарь для хранения метрик
    all_metrics = {}
    
    # Для каждого алгоритма строим график
    for algo_name, data in best_solutions.items():
        Kp, Ki, Kd = data['params']
        fitness = data['fitness']
        
        # Получаем отклик системы
        t_out, y_out = get_motor_response(Kp, Ki, Kd, t)
        
        # Вычисляем метрики
        overshoot, rise_time, settling_time = calculate_step_metrics(t_out, y_out)
        all_metrics[algo_name] = {
            'overshoot': overshoot,
            'rise_time': rise_time,
            'settling_time': settling_time,
            'fitness': fitness,
            'Kp': Kp, 'Ki': Ki, 'Kd': Kd
        }
        
        # Строим графики
        color = colors.get(algo_name, 'gray')
        label = f"{algo_name} (фитнес={fitness:.2e})"
        
        ax1.plot(t_out, y_out, label=label, color=color, linewidth=2)
        ax2.plot(t_out, y_out, color=color, linewidth=2)
    
    # Добавляем легенду
    ax1.legend(loc='lower right', fontsize=9)
    
    # Создаем текстовую таблицу с метриками
    metrics_text = "Метрики переходных процессов:\n"
    metrics_text += "-" * 70 + "\n"
    metrics_text += f"{'Алгоритм':<8} {'Перерег. %':<12} {'Время нараст. (с)':<16} {'Время устан. (с)':<16}\n"
    metrics_text += "-" * 70 + "\n"
    
    for algo_name, metrics in all_metrics.items():
        overshoot_str = f"{metrics['overshoot']:.2f}" if metrics['overshoot'] < 1000 else "∞"
        rise_str = f"{metrics['rise_time']:.3f}" if metrics['rise_time'] < 1000 else "∞"
        settle_str = f"{metrics['settling_time']:.3f}" if metrics['settling_time'] < 1000 else "∞"
        
        metrics_text += f"{algo_name:<8} {overshoot_str:<12} {rise_str:<16} {settle_str:<16}\n"
    
    # Добавляем текст под графиком
    fig.text(0.5, 0.01, metrics_text, ha='center', fontsize=9, family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)  # Оставляем место для текста с метриками
    
    # Сохраняем график
    output_file = "step_response_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✅ График переходных процессов сохранен: {output_file}")
    
    # Показываем график
    plt.show()
    
    return all_metrics

def create_metrics_table(metrics):
    """Создает CSV-файл с метриками переходных процессов"""
    
    if not metrics:
        return
    
    # Создаем DataFrame
    data = []
    for algo_name, m in metrics.items():
        data.append({
            'Algorithm': algo_name,
            'Kp': m['Kp'],
            'Ki': m['Ki'],
            'Kd': m['Kd'],
            'Fitness': m['fitness'],
            'Overshoot_%': m['overshoot'],
            'Rise_Time_s': m['rise_time'],
            'Settling_Time_s': m['settling_time']
        })
    
    df = pd.DataFrame(data)
    
    # Сортируем по фитнесу
    df = df.sort_values('Fitness')
    
    # Сохраняем в CSV
    output_file = "step_response_metrics.csv"
    df.to_csv(output_file, index=False, float_format='%.6f')
    print(f"✅ Метрики сохранены: {output_file}")
    
    # Выводим таблицу в консоль
    print("\n" + "="*80)
    print("📊 МЕТРИКИ ПЕРЕХОДНЫХ ПРОЦЕССОВ")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)
    
    return df

def compare_with_nature_paper(metrics):
    """Сравнивает результаты с показателями из статьи Nature"""
    
    print("\n" + "="*80)
    print("📚 СРАВНЕНИЕ С РЕЗУЛЬТАТАМИ ИЗ СТАТЬИ NATURE")
    print("="*80)
    print("Статья: 'Advanced control parameter optimization in DC motors'")
    print("Показатели MGO-PID (из статьи):")
    print("  • Время нарастания: 0.0478 с")
    print("  • Перерегулирование: 0%")
    print("  • Время установления: 0.0841 с")
    print("-" * 60)
    
    print("\nНаши лучшие результаты:")
    best_algo = min(metrics.items(), key=lambda x: x[1]['fitness'])
    algo_name, m = best_algo
    
    print(f"  • Лучший алгоритм: {algo_name}")
    print(f"  • Время нарастания: {m['rise_time']:.4f} с")
    print(f"  • Перерегулирование: {m['overshoot']:.2f}%")
    print(f"  • Время установления: {m['settling_time']:.4f} с")
    print(f"  • Фитнес (ITAE): {m['fitness']:.6f}")
    print("-" * 60)
    
    if m['rise_time'] < 0.05:
        print("✓ Наше время нарастания сопоставимо с лучшими мировыми показателями!")
    else:
        print("ℹ Наше время нарастания можно улучшить дальнейшей оптимизацией")

def main():
    """Основная функция"""
    
    print("=" * 80)
    print("🔄 ПОСТРОЕНИЕ ГРАФИКОВ ПЕРЕХОДНЫХ ПРОЦЕССОВ")
    print("=" * 80)
    
    # Загружаем лучшие решения
    print("\n🔍 Загрузка лучших решений из результатов эксперимента...")
    best_solutions = load_best_solutions()
    
    if not best_solutions:
        print("\n❌ Не удалось загрузить решения. Убедитесь, что эксперименты выполнены.")
        print("   Сначала запустите: python experiment.py")
        return
    
    # Строим графики переходных процессов
    print("\n📈 Построение графиков...")
    metrics = plot_step_responses(best_solutions)
    
    if metrics:
        # Создаем таблицу с метриками
        create_metrics_table(metrics)
        
        # Сравниваем со статьей Nature
        compare_with_nature_paper(metrics)
    
    print("\n" + "=" * 80)
    print("✅ ГОТОВО! Файлы созданы:")
    print("   • step_response_comparison.png - график переходных процессов")
    print("   • step_response_metrics.csv - таблица с метриками")
    print("=" * 80)

if __name__ == "__main__":
    main()