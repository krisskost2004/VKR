"""
Модуль для визуализации результатов экспериментов.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os

# Настройка стиля графиков
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.figsize': (12, 8),
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

# Названия задач для отображения
problem_titles = {
    'dc_motor_pid': 'ПИД-регулятор двигателя',
    'inverted_pendulum': 'Балансировка маятника',
    'liquid_level': 'Уровень жидкости'
}

# Цвета для алгоритмов
colors = {
    'PSO': '#FF6B6B',    # Красный
    'GWO': '#4ECDC4',    # Бирюзовый
    'WOA': '#45B7D1',    # Голубой
    'HHO': '#96CEB4',    # Зеленый
    'SMA': '#FFEAA7'     # Желтый
}

def load_results():
    """Загрузка результатов из файлов"""
    
    results = {}
    
    # Загрузка сводных результатов
    summary_file = "experiment_results/summary_results.csv"
    if os.path.exists(summary_file):
        results['summary'] = pd.read_csv(summary_file)
        print(f"Загружены сводные результаты из: {summary_file}")
    else:
        print(f"Файл {summary_file} не найден")
    
    # Загрузка данных сходимости
    convergence_file = "experiment_results/convergence.json"
    if os.path.exists(convergence_file):
        with open(convergence_file, 'r') as f:
            results['convergence'] = json.load(f)
        print(f"Загружены данные сходимости из: {convergence_file}")
    else:
        print(f"Файл {convergence_file} не найден")
    
    return results

def plot_convergence(results, save_dir="plots"):
    """Построение графиков сходимости"""
    
    if 'convergence' not in results:
        print("Данные сходимости не загружены")
        return
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Построение графиков для каждой задачи
    for problem_name, algorithms_data in results['convergence'].items():
        plt.figure(figsize=(10, 6))
        
        for algo_name, history in algorithms_data.items():
            if history and len(history) > 0:
                plt.plot(history, 
                        label=algo_name,
                        color=colors.get(algo_name, 'gray'),
                        linewidth=2,
                        alpha=0.8)
        
        plt.title(f'{problem_titles.get(problem_name, problem_name)}', fontsize=14)
        plt.xlabel('Итерация', fontsize=12)
        plt.ylabel('Значение целевой функции', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')  # Логарифмическая шкала для лучшей визуализации
        
        # Сохранение графика
        filename = f"convergence_{problem_name}.png"
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"График сходимости сохранен: {filepath}")
        plt.close()

def plot_quality_speed_comparison(save_dir="plots"):
    """
    Создает столбчатую диаграмму, где для каждой задачи и алгоритма
    отображаются два столбца: качество (фитнес) и скорость (время).
    """
    
    # Загружаем сводные результаты
    summary_file = os.path.join("experiment_results", "summary_results.csv")
    if not os.path.exists(summary_file):
        print(f"Файл {summary_file} не найден")
        return
    
    df = pd.read_csv(summary_file)
    os.makedirs(save_dir, exist_ok=True)
    
    # Нормализуем данные для отображения на одном графике
    # (чтобы и фитнес, и время были в сопоставимом масштабе 0-1)
    df_normalized = df.copy()
    
    for problem in df['Problem'].unique():
        problem_mask = df['Problem'] == problem
        
        # Нормализация фитнеса (меньше - лучше)
        fitness_col = 'Best_Fitness_Mean'
        fitness_min = df.loc[problem_mask, fitness_col].min()
        fitness_max = df.loc[problem_mask, fitness_col].max()
        
        if fitness_max > fitness_min:
            df_normalized.loc[problem_mask, 'Fitness_Norm'] = 1 - (
                (df.loc[problem_mask, fitness_col] - fitness_min) / 
                (fitness_max - fitness_min)
            )
        else:
            df_normalized.loc[problem_mask, 'Fitness_Norm'] = 0.5
        
        # Нормализация времени (меньше - лучше)
        time_col = 'Execution_Time_Mean'
        time_min = df.loc[problem_mask, time_col].min()
        time_max = df.loc[problem_mask, time_col].max()
        
        if time_max > time_min:
            df_normalized.loc[problem_mask, 'Time_Norm'] = 1 - (
                (df.loc[problem_mask, time_col] - time_min) / 
                (time_max - time_min)
            )
        else:
            df_normalized.loc[problem_mask, 'Time_Norm'] = 0.5
    
    # Создаем фигуру с подграфиками для каждой задачи
    problems = df['Problem'].unique()
    n_problems = len(problems)
    
    fig, axes = plt.subplots(1, n_problems, figsize=(6*n_problems, 6))
    if n_problems == 1:
        axes = [axes]
    
    fig.suptitle('Сравнение качества и скорости работы алгоритмов\n(нормированные значения: 1 - лучше, 0 - хуже)', 
                fontsize=14, fontweight='bold', y=1.05)
    
    for idx, problem in enumerate(problems):
        ax = axes[idx]
        problem_data = df_normalized[df_normalized['Problem'] == problem]
        problem_data_orig = df[df['Problem'] == problem]
        algorithms = problem_data['Algorithm'].values
        
        x = np.arange(len(algorithms))
        width = 0.35  # ширина столбцов
        
        # Столбцы для качества (фитнес)
        fitness_bars = ax.bar(x - width/2, 
                             problem_data['Fitness_Norm'].values,
                             width, 
                             label='Качество (фитнес)',
                             color='skyblue',
                             edgecolor='navy',
                             alpha=0.7,
                             linewidth=1)
        
        # Столбцы для скорости (время)
        time_bars = ax.bar(x + width/2,
                          problem_data['Time_Norm'].values,
                          width,
                          label='Скорость (время)',
                          color='lightcoral',
                          edgecolor='darkred',
                          alpha=0.7,
                          linewidth=1)
        
        # Добавляем фактические значения на столбцы
        for i, (fitness_bar, time_bar) in enumerate(zip(fitness_bars, time_bars)):
            # Значение фитнеса
            fitness_val = problem_data_orig['Best_Fitness_Mean'].values[i]
            fitness_text = f'{fitness_val:.2e}'
            if fitness_val > 1e5:  # Для больших значений (штрафы)
                fitness_text = f'{fitness_val:.0e}'
            
            ax.text(fitness_bar.get_x() + fitness_bar.get_width()/2, 
                   fitness_bar.get_height() + 0.02,
                   fitness_text,
                   ha='center', va='bottom', fontsize=8, rotation=45)
            
            # Значение времени
            time_val = problem_data_orig['Execution_Time_Mean'].values[i]
            ax.text(time_bar.get_x() + time_bar.get_width()/2,
                   time_bar.get_height() + 0.02,
                   f'{time_val:.2f}с',
                   ha='center', va='bottom', fontsize=8, rotation=45)
        
        # Настройка осей
        ax.set_title(problem_titles.get(problem, problem), fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(algorithms)
        ax.set_ylabel('Нормированное значение (1 = лучший)')
        ax.set_ylim(0, 1.2)  # Оставляем место для подписей
        ax.grid(True, alpha=0.3, axis='y')
        
        # Добавляем легенду только для первого графика
        if idx == 0:
            ax.legend(loc='upper right')
        
        # Добавляем цветовые метки алгоритмов
        for i, algo in enumerate(algorithms):
            ax.get_xticklabels()[i].set_color(colors.get(algo, 'black'))
    
    plt.tight_layout()
    
    # Сохраняем график
    output_file = os.path.join(save_dir, "quality_speed_comparison.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"График сравнения качества и скорости сохранен: {output_file}")
    plt.close()
    
    return fig

def create_detailed_ranking_table(save_dir="plots"):
    """
    Создает детальную таблицу с ранжированием алгоритмов
    по качеству и скорости для каждой задачи.
    """
    
    # Загружаем данные
    summary_file = os.path.join("experiment_results", "summary_results.csv")
    if not os.path.exists(summary_file):
        print(f"Файл {summary_file} не найден")
        return
    
    df = pd.read_csv(summary_file)
    os.makedirs(save_dir, exist_ok=True)
    
    # Создаем фигуру для таблицы
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.axis('off')
    ax.axis('tight')
    
    # Подготавливаем данные для таблицы
    table_data = []
    headers = ['Задача', 'Алгоритм', 'Качество (фитнес)', 'Ранг\nкачества', 
               'Время (с)', 'Ранг\nскорости', 'Ср. ранг', 'Лучший?']
    
    all_best = []
    
    for problem in df['Problem'].unique():
        problem_df = df[df['Problem'] == problem].copy()
        
        # Вычисляем ранги
        problem_df.loc[:, 'Quality_Rank'] = problem_df['Best_Fitness_Mean'].rank(method='min')
        problem_df.loc[:, 'Speed_Rank'] = problem_df['Execution_Time_Mean'].rank(method='min')
        problem_df.loc[:, 'Avg_Rank'] = (problem_df['Quality_Rank'] + problem_df['Speed_Rank']) / 2
        
        # Определяем лучший алгоритм (минимальный средний ранг)
        best_avg_rank = problem_df['Avg_Rank'].min()
        problem_df.loc[:, 'Is_Best'] = problem_df['Avg_Rank'] == best_avg_rank
        
        # Сортируем по среднему рангу
        problem_df = problem_df.sort_values('Avg_Rank')
        
        for _, row in problem_df.iterrows():
            is_best = row['Is_Best']
            best_marker = '✓' if is_best else ''
            if is_best:
                all_best.append(row['Algorithm'])
            
            # Форматируем значение фитнеса
            fitness_val = row['Best_Fitness_Mean']
            if fitness_val > 1e5:
                fitness_str = f'{fitness_val:.0e}'
            else:
                fitness_str = f'{fitness_val:.4e}'
            
            table_data.append([
                problem_titles.get(row['Problem'], row['Problem']),
                row['Algorithm'],
                fitness_str,
                f"{int(row['Quality_Rank'])}",
                f"{row['Execution_Time_Mean']:.3f}",
                f"{int(row['Speed_Rank'])}",
                f"{row['Avg_Rank']:.1f}",
                best_marker
            ])
        
        # Добавляем разделитель между задачами
        table_data.append(['---', '---', '---', '---', '---', '---', '---', '---'])
    
    # Создаем таблицу
    table = ax.table(cellText=table_data, colLabels=headers,
                    cellLoc='center', loc='center',
                    colWidths=[0.25, 0.1, 0.15, 0.08, 0.08, 0.08, 0.08, 0.08])
    
    # Стилизация таблицы
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    # Выделяем заголовок
    for i, header in enumerate(headers):
        cell = table[0, i]
        cell.set_facecolor('#4472C4')
        cell.set_text_props(weight='bold', color='white')
    
    # Раскрашиваем строки и выделяем лучшие
    for i in range(1, len(table_data)):
        row_data = table_data[i-1]
        
        # Пропускаем разделители
        if row_data[0] == '---':
            for j in range(len(headers)):
                cell = table[i, j]
                cell.set_facecolor('#E0E0E0')
            continue
        
        # Проверяем, лучший ли это алгоритм
        is_best = row_data[7] == '✓'
        
        for j in range(len(headers)):
            cell = table[i, j]
            
            # Цвет фона для строк
            if i % 2 == 0:
                cell.set_facecolor('#F2F2F2')
            
            # Выделяем лучший алгоритм жирным
            if is_best:
                cell.set_text_props(weight='bold')
                
                # Добавляем цветной фон для столбца с алгоритмом
                if j == 1:  # Столбец с названием алгоритма
                    algo = row_data[1]
                    if algo in colors:
                        # Делаем полупрозрачный фон
                        rgba = plt.matplotlib.colors.to_rgba(colors[algo], alpha=0.3)
                        cell.set_facecolor(rgba)
    
    ax.set_title('Детальный рейтинг алгоритмов по качеству и скорости\n(меньший ранг = лучше)', 
                fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    # Сохраняем
    output_file = os.path.join(save_dir, "quality_speed_ranking.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Таблица рейтингов сохранена: {output_file}")
    plt.close()
    
    # Статистика по лучшим алгоритмам
    from collections import Counter
    best_counter = Counter(all_best)
    
    print("\n" + "="*60)
    print("СТАТИСТИКА ЛУЧШИХ АЛГОРИТМОВ")
    print("="*60)
    for algo, count in best_counter.most_common():
        print(f"{algo}: лучший в {count} задаче(ах)")
    print("="*60)

def plot_radar_chart(save_dir="plots"):
    """
    Создает радарную диаграмму для сравнения алгоритмов по нескольким метрикам.
    """
    
    # Загружаем данные
    summary_file = os.path.join("experiment_results", "summary_results.csv")
    if not os.path.exists(summary_file):
        print(f"Файл {summary_file} не найден")
        return
    
    df = pd.read_csv(summary_file)
    os.makedirs(save_dir, exist_ok=True)
    
    # Вычисляем средние метрики по всем задачам для каждого алгоритма
    algorithms = df['Algorithm'].unique()
    
    metrics = {}
    for algo in algorithms:
        algo_df = df[df['Algorithm'] == algo]
        
        # Средний фитнес
        avg_fitness = algo_df['Best_Fitness_Mean'].mean()
        # Среднее время
        avg_time = algo_df['Execution_Time_Mean'].mean()
        # Успешность (если есть)
        success_rate = algo_df['Success_Rate_%'].mean() if 'Success_Rate_%' in algo_df.columns else 50
        # Стабильность (обратная величина std)
        avg_std = algo_df['Best_Fitness_Std'].mean()
        stability = 1 / (1 + avg_std)  # Нормализуем
        
        metrics[algo] = {
            'fitness': avg_fitness,
            'time': avg_time,
            'success': success_rate,
            'stability': stability
        }
    
    # Нормализуем метрики для радарной диаграммы
    normalized = {}
    for metric in ['fitness', 'time', 'success', 'stability']:
        values = [metrics[algo][metric] for algo in algorithms]
        min_val, max_val = min(values), max(values)
        
        for algo in algorithms:
            if algo not in normalized:
                normalized[algo] = {}
            
            if metric in ['fitness', 'time']:  # Меньше = лучше
                if max_val > min_val:
                    norm_val = 1 - (metrics[algo][metric] - min_val) / (max_val - min_val)
                else:
                    norm_val = 0.5
            else:  # Больше = лучше
                if max_val > min_val:
                    norm_val = (metrics[algo][metric] - min_val) / (max_val - min_val)
                else:
                    norm_val = 0.5
            
            normalized[algo][metric] = norm_val
    
    # Создаем радарную диаграмму
    categories = ['Качество\n(фитнес)', 'Скорость\n(время)', 'Успешность\n(%)', 'Стабильность']
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Замыкаем круг
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    for algo in algorithms:
        values = [
            normalized[algo]['fitness'],
            normalized[algo]['time'],
            normalized[algo]['success'] / 100,  # Нормализуем успешность
            normalized[algo]['stability']
        ]
        values += values[:1]  # Замыкаем круг
        
        ax.plot(angles, values, 'o-', linewidth=2, label=algo, color=colors.get(algo, 'gray'))
        ax.fill(angles, values, alpha=0.1, color=colors.get(algo, 'gray'))
    
    # Настройка внешнего вида
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_title('Сравнение алгоритмов по комплексным метрикам', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.grid(True)
    
    # Сохраняем
    output_file = os.path.join(save_dir, "radar_chart.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Радарная диаграмма сохранена: {output_file}")
    plt.close()

def plot_performance_comparison(save_dir="plots"):
    """
    Построение базовых сравнительных графиков (для обратной совместимости)
    """
    
    if not os.path.exists("experiment_results/summary_results.csv"):
        print("Сводные результаты не загружены")
        return
    
    df = pd.read_csv("experiment_results/summary_results.csv")
    os.makedirs(save_dir, exist_ok=True)
    
    # График 1: Сравнение качества решения
    plt.figure(figsize=(14, 8))
    
    algorithms = df['Algorithm'].unique()
    problems = df['Problem'].unique()
    
    # Создаем подграфики
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # График качества решения
    ax1 = axes[0, 0]
    x = np.arange(len(problems))
    width = 0.15
    
    for i, algo in enumerate(algorithms):
        algo_data = df[df['Algorithm'] == algo]
        values = []
        errors = []
        
        for problem in problems:
            problem_data = algo_data[algo_data['Problem'] == problem]
            if not problem_data.empty:
                values.append(problem_data['Best_Fitness_Mean'].values[0])
                errors.append(problem_data['Best_Fitness_Std'].values[0])
            else:
                values.append(0)
                errors.append(0)
        
        ax1.bar(x + i*width, values, yerr=errors, width=width, label=algo,
               color=colors.get(algo, 'gray'), alpha=0.8, capsize=3,
               error_kw={'elinewidth': 1})
    
    ax1.set_xlabel('Задача', fontsize=12)
    ax1.set_ylabel('Значение фитнеса', fontsize=12)
    ax1.set_title('Качество решения (меньше - лучше)', fontsize=14)
    ax1.set_xticks(x + width*2)
    ax1.set_xticklabels([problem_titles.get(p, p) for p in problems], rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_yscale('log')
    
    # График времени выполнения
    ax2 = axes[0, 1]
    
    for i, algo in enumerate(algorithms):
        algo_data = df[df['Algorithm'] == algo]
        values = []
        errors = []
        
        for problem in problems:
            problem_data = algo_data[algo_data['Problem'] == problem]
            if not problem_data.empty:
                values.append(problem_data['Execution_Time_Mean'].values[0])
                errors.append(problem_data['Execution_Time_Std'].values[0])
            else:
                values.append(0)
                errors.append(0)
        
        ax2.bar(x + i*width, values, yerr=errors, width=width, label=algo,
               color=colors.get(algo, 'gray'), alpha=0.8, capsize=3,
               error_kw={'elinewidth': 1})
    
    ax2.set_xlabel('Задача', fontsize=12)
    ax2.set_ylabel('Время выполнения (секунды)', fontsize=12)
    ax2.set_title('Время выполнения', fontsize=14)
    ax2.set_xticks(x + width*2)
    ax2.set_xticklabels([problem_titles.get(p, p) for p in problems], rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # График успешности
    ax3 = axes[1, 0]
    
    if 'Success_Rate_%' in df.columns:
        for i, algo in enumerate(algorithms):
            algo_data = df[df['Algorithm'] == algo]
            values = []
            
            for problem in problems:
                problem_data = algo_data[algo_data['Problem'] == problem]
                if not problem_data.empty and 'Success_Rate_%' in problem_data.columns:
                    values.append(problem_data['Success_Rate_%'].values[0])
                else:
                    values.append(0)
            
            ax3.bar(x + i*width, values, width=width, label=algo,
                   color=colors.get(algo, 'gray'), alpha=0.8)
        
        ax3.set_xlabel('Задача', fontsize=12)
        ax3.set_ylabel('Успешность (%)', fontsize=12)
        ax3.set_title('Процент успешных запусков', fontsize=14)
        ax3.set_xticks(x + width*2)
        ax3.set_xticklabels([problem_titles.get(p, p) for p in problems], rotation=45, ha='right')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.set_ylim(0, 105)
    else:
        ax3.text(0.5, 0.5, 'Нет данных об успешности', 
                ha='center', va='center', transform=ax3.transAxes)
    
    # График 4: Рейтинг алгоритмов
    ax4 = axes[1, 1]
    
    # Вычисляем средний рейтинг
    algo_ranking = []
    for algo in algorithms:
        algo_df = df[df['Algorithm'] == algo]
        avg_fitness = algo_df['Best_Fitness_Mean'].mean()
        avg_time = algo_df['Execution_Time_Mean'].mean()
        
        # Комбинированный рейтинг (меньше - лучше)
        # Нормализуем
        fitness_rank = (avg_fitness - df['Best_Fitness_Mean'].min()) / \
                      (df['Best_Fitness_Mean'].max() - df['Best_Fitness_Mean'].min() + 1e-10)
        time_rank = (avg_time - df['Execution_Time_Mean'].min()) / \
                   (df['Execution_Time_Mean'].max() - df['Execution_Time_Mean'].min() + 1e-10)
        
        combined_rank = (fitness_rank + time_rank) / 2
        algo_ranking.append((algo, combined_rank))
    
    # Сортируем по рейтингу (меньше = лучше)
    algo_ranking.sort(key=lambda x: x[1])
    sorted_algos = [a[0] for a in algo_ranking]
    sorted_ranks = [a[1] for a in algo_ranking]
    
    # Инвертируем для наглядности (больше = лучше)
    display_scores = [1 - r for r in sorted_ranks]
    
    bars = ax4.barh(sorted_algos, display_scores, 
                    color=[colors.get(a, 'gray') for a in sorted_algos],
                    alpha=0.8)
    
    ax4.set_xlabel('Комбинированный рейтинг (больше - лучше)', fontsize=12)
    ax4.set_title('Общий рейтинг алгоритмов', fontsize=14)
    ax4.grid(True, alpha=0.3, axis='x')
    ax4.set_xlim(0, 1)
    
    # Добавляем значения на столбцы
    for bar, score in zip(bars, display_scores):
        ax4.text(score + 0.02, bar.get_y() + bar.get_height()/2,
                f'{score:.2f}', va='center', fontsize=10)
    
    plt.tight_layout()
    
    # Сохранение графика
    filename = "performance_comparison.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"Сравнительный график сохранен: {filepath}")
    plt.close()

def main():
    """Основная функция визуализации"""
    
    print("=" * 80)
    print("ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ")
    print("=" * 80)
    
    # Создаем директорию для графиков
    plots_dir = "plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    # Загружаем результаты
    results = load_results()
    
    if not results:
        print("Нет данных для визуализации")
        return
    
    # Создаем базовые графики
    print("\n1. Создание графиков сходимости...")
    plot_convergence(results, plots_dir)
    
    print("\n2. Создание базовых сравнительных графиков...")
    plot_performance_comparison(plots_dir)
    
    print("\n3. Создание улучшенных сравнительных графиков...")
    plot_quality_speed_comparison(plots_dir)
    
    print("\n4. Создание детальной таблицы рейтингов...")
    create_detailed_ranking_table(plots_dir)
    
    print("\n5. Создание радарной диаграммы...")
    plot_radar_chart(plots_dir)
    
    print("\n" + "=" * 80)
    print("ВИЗУАЛИЗАЦИЯ ЗАВЕРШЕНА")
    print(f"Все графики сохранены в директории: {plots_dir}")
    print("=" * 80)

if __name__ == "__main__":
    main()