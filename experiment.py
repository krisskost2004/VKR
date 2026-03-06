"""
Основной скрипт для проведения сравнительных экспериментов.
Запускает все алгоритмы на всех задачах и собирает метрики.
Автоматически запускает визуализацию после завершения.
"""

import numpy as np
import pandas as pd
import json
import time
from datetime import datetime
import os
import subprocess
import sys
from typing import Dict, List, Tuple, Any

from algorithms import PSO, GWO, WOA, HHO, SMA
from problems import get_problem_info

class ComparativeExperiment:
    """Класс для проведения сравнительных экспериментов"""
    
    def __init__(self, 
                 num_runs: int = 10,
                 max_iter: int = 100,
                 pop_size: int = 30,
                 output_dir: str = "experiment_results",
                 auto_visualize: bool = True):
        """
        Args:
            num_runs: Количество независимых запусков каждого алгоритма
            max_iter: Максимальное количество итераций
            pop_size: Размер популяции
            output_dir: Директория для сохранения результатов
            auto_visualize: Автоматически запускать визуализацию после экспериментов
        """
        self.num_runs = num_runs
        self.max_iter = max_iter
        self.pop_size = pop_size
        self.output_dir = output_dir
        self.auto_visualize = auto_visualize
        
        # Создаем директорию для результатов
        os.makedirs(output_dir, exist_ok=True)
        
        # Определяем алгоритмы для сравнения
        self.algorithms = {
            'PSO': PSO,
            'GWO': GWO,
            'WOA': WOA,
            'HHO': HHO,
            'SMA': SMA
        }
        
        # Определяем задачи для тестирования
        self.problems = {
            'dc_motor_pid': 'Оптимизация ПИД-регулятора (двигатель)',
            'inverted_pendulum': 'Балансировка маятника',
            'liquid_level': 'Управление уровнем жидкости'
        }
        
        # Названия задач для красивого вывода
        self.problem_titles = {
            'dc_motor_pid': 'ПИД-регулятор двигателя',
            'inverted_pendulum': 'Балансировка маятника',
            'liquid_level': 'Уровень жидкости'
        }
        
        # Результаты экспериментов
        self.results = {}
        self.convergence_data = {}
        
        # Время начала эксперимента
        self.start_time = None
        
    def run_single_experiment(self, 
                             algorithm_class,
                             problem_info: Dict,
                             algorithm_name: str,
                             problem_name: str,
                             run_id: int) -> Dict[str, Any]:
        """
        Запуск одного эксперимента (один алгоритм на одной задаче).
        """
        try:
            # Фиксируем seed для воспроизводимости
            seed = 42 + run_id * 100
            
            # Извлекаем информацию о задаче
            dim = problem_info['dim']
            bounds = problem_info['bounds']
            objective_func = problem_info['objective_func']
            
            # Создаем экземпляр алгоритма с оптимальными параметрами
            if algorithm_name == 'PSO':
                algorithm = algorithm_class(
                    objective_func=objective_func,
                    dim=dim,
                    bounds=bounds,
                    max_iter=self.max_iter,
                    pop_size=self.pop_size,
                    w=0.7,
                    c1=1.5,
                    c2=1.5,
                    seed=seed
                )
            elif algorithm_name == 'SMA':
                algorithm = algorithm_class(
                    objective_func=objective_func,
                    dim=dim,
                    bounds=bounds,
                    max_iter=self.max_iter,
                    pop_size=self.pop_size,
                    z=0.03,
                    seed=seed
                )
            else:
                algorithm = algorithm_class(
                    objective_func=objective_func,
                    dim=dim,
                    bounds=bounds,
                    max_iter=self.max_iter,
                    pop_size=self.pop_size,
                    seed=seed
                )
            
            # Запускаем оптимизацию
            best_solution, best_fitness = algorithm.optimize()
            
            # Получаем метрики
            metrics = algorithm.get_metrics()
            
            # Дополнительные метрики
            metrics.update({
                'algorithm': algorithm_name,
                'problem': problem_name,
                'run_id': run_id,
                'seed': seed,
                'solution': best_solution.tolist(),
                'best_fitness': float(best_fitness)
            })
            
            return metrics
            
        except Exception as e:
            print(f"   Ошибка в {algorithm_name}: {str(e)[:100]}...")
            return {
                'algorithm': algorithm_name,
                'problem': problem_name,
                'run_id': run_id,
                'best_fitness': float('inf'),
                'execution_time': 0,
                'function_evaluations': 0,
                'error': str(e)
            }
    
    def print_progress_bar(self, current, total, bar_length=50):
        """Печатает прогресс-бар"""
        percent = current / total
        arrow = '=' * int(round(percent * bar_length))
        spaces = ' ' * (bar_length - len(arrow))
        
        elapsed = time.time() - self.start_time
        if current > 0:
            eta = (elapsed / current) * (total - current)
            eta_str = time.strftime("%H:%M:%S", time.gmtime(eta))
        else:
            eta_str = "??:??:??"
        
        elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
        
        sys.stdout.write(f"\r[{arrow}{spaces}] {current}/{total} "
                        f"({percent*100:.1f}%) | "
                        f"Прошло: {elapsed_str} | Осталось: {eta_str}")
        sys.stdout.flush()
    
    def run_all_experiments(self):
        """Запуск всех экспериментов"""
        self.start_time = time.time()
        
        print("=" * 90)
        print("  ЗАПУСК СРАВНИТЕЛЬНЫХ ЭКСПЕРИМЕНТОВ")
        print("=" * 90)
        print(f" Дата и время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f" Количество запусков: {self.num_runs}")
        print(f" Максимальное количество итераций: {self.max_iter}")
        print(f" Размер популяции: {self.pop_size}")
        print(f" Директория результатов: {self.output_dir}")
        print("=" * 90)
        
        total_experiments = len(self.algorithms) * len(self.problems) * self.num_runs
        current_experiment = 0
        
        # Словарь для хранения времени выполнения по задачам
        problem_times = {}
        
        # Цикл по всем задачам
        for problem_idx, (problem_name, problem_description) in enumerate(self.problems.items(), 1):
            problem_start = time.time()
            
            print(f"\n{'='*60}")
            print(f" ЗАДАЧА {problem_idx}/{len(self.problems)}: {problem_description}")
            print(f"{'='*60}")
            
            # Получаем информацию о задаче
            problem_info = get_problem_info(problem_name)
            if problem_info is None:
                print(f"❌ Ошибка: задача '{problem_name}' не найдена")
                continue
            
            # Инициализируем структуры для хранения результатов по задаче
            self.results[problem_name] = {}
            self.convergence_data[problem_name] = {}
            
            # Цикл по всем алгоритмам
            for algo_idx, (algorithm_name, algorithm_class) in enumerate(self.algorithms.items(), 1):
                print(f"\n  Алгоритм {algo_idx}/{len(self.algorithms)}: {algorithm_name}")
                print("  " + "-" * 40)
                
                # Массивы для хранения результатов по запускам
                all_metrics = []
                convergence_histories = []
                
                # Множественные запуски для статистической значимости
                for run_id in range(self.num_runs):
                    current_experiment += 1
                    self.print_progress_bar(current_experiment, total_experiments)
                    
                    # Запуск эксперимента
                    metrics = self.run_single_experiment(
                        algorithm_class=algorithm_class,
                        problem_info=problem_info,
                        algorithm_name=algorithm_name,
                        problem_name=problem_name,
                        run_id=run_id
                    )
                    
                    # Сохраняем результаты
                    all_metrics.append(metrics)
                    
                    # Сохраняем историю сходимости (только для успешных запусков)
                    if run_id == 0 and 'convergence_history' in metrics:
                        convergence_histories = metrics['convergence_history']
                        self.convergence_data[problem_name][algorithm_name] = convergence_histories
                
                # Очищаем строку прогресс-бара
                print()
                
                # Фильтруем успешные запуски
                successful_runs = [m for m in all_metrics if 'error' not in m and m['best_fitness'] < 1e9]
                
                if successful_runs:
                    best_fitness_values = [m['best_fitness'] for m in successful_runs]
                    execution_times = [m['execution_time'] for m in successful_runs]
                    
                    mean_fitness = np.mean(best_fitness_values)
                    std_fitness = np.std(best_fitness_values)
                    mean_time = np.mean(execution_times)
                    std_time = np.std(execution_times)
                    
                    # Цветной вывод в зависимости от качества
                    if mean_fitness < 1e-1:
                        status = "✅"
                    elif mean_fitness < 1e6:
                        status = "⚠️"
                    else:
                        status = "❌"
                    
                    print(f"  {status} {algorithm_name}: "
                          f"Фитнес = {mean_fitness:.4e} ± {std_fitness:.4e}, "
                          f"Время = {mean_time:.3f} ± {std_time:.3f} с")
                    
                    # Сохраняем агрегированные результаты
                    self.results[problem_name][algorithm_name] = {
                        'best_fitness_mean': float(mean_fitness),
                        'best_fitness_std': float(std_fitness),
                        'best_fitness_min': float(np.min(best_fitness_values)),
                        'best_fitness_max': float(np.max(best_fitness_values)),
                        'execution_time_mean': float(mean_time),
                        'execution_time_std': float(std_time),
                        'all_runs': all_metrics,
                        'convergence_history': convergence_histories,
                        'success_rate': len(successful_runs) / len(all_metrics) * 100
                    }
                    
                else:
                    print(f"  ❌ {algorithm_name}: Нет успешных запусков")
                    self.results[problem_name][algorithm_name] = {
                        'error': 'Нет успешных запусков',
                        'all_runs': all_metrics
                    }
            
            problem_time = time.time() - problem_start
            problem_times[problem_name] = problem_time
            print(f"\n   Время выполнения задачи: {problem_time:.2f} с")
        
        # Сохраняем все результаты
        self.save_results()
        
        # Выводим итоговую статистику
        self.print_final_statistics(problem_times)
        
        # Автоматически запускаем визуализацию
        if self.auto_visualize:
            self.run_visualization()
        
        return self.results
    
    def save_results(self):
        """Сохранение всех результатов в файлы"""
        
        print("\n" + "=" * 60)
        print("СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
        print("=" * 60)
        
        # Сохраняем агрегированные результаты в CSV
        summary_data = []
        
        for problem_name, algorithms in self.results.items():
            for algorithm_name, metrics in algorithms.items():
                if 'error' not in metrics:
                    summary_data.append({
                        'Problem': self.problem_titles.get(problem_name, problem_name),
                        'Algorithm': algorithm_name,
                        'Best_Fitness_Mean': metrics['best_fitness_mean'],
                        'Best_Fitness_Std': metrics['best_fitness_std'],
                        'Execution_Time_Mean': metrics['execution_time_mean'],
                        'Execution_Time_Std': metrics['execution_time_std'],
                        'Success_Rate_%': metrics.get('success_rate', 0)
                    })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_path = os.path.join(self.output_dir, "summary_results.csv")
            summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
            print(f" Сводные результаты: {summary_path}")
        
        # Сохраняем детальные результаты в JSON
        detailed_path = os.path.join(self.output_dir, "results.json")
        with open(detailed_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, default=str, ensure_ascii=False)
        print(f" Детальные результаты: {detailed_path}")
        
        # Сохраняем данные сходимости
        convergence_path = os.path.join(self.output_dir, "convergence.json")
        with open(convergence_path, 'w', encoding='utf-8') as f:
            json.dump(self.convergence_data, f, indent=2, default=str, ensure_ascii=False)
        print(f" Данные сходимости: {convergence_path}")
        
        # Сохраняем параметры эксперимента
        experiment_params = {
            'num_runs': self.num_runs,
            'max_iter': self.max_iter,
            'pop_size': self.pop_size,
            'timestamp': datetime.now().isoformat(),
            'total_time': time.time() - self.start_time
        }
        
        params_path = os.path.join(self.output_dir, "params.json")
        with open(params_path, 'w') as f:
            json.dump(experiment_params, f, indent=2)
        print(f" Параметры эксперимента: {params_path}")
    
    def print_final_statistics(self, problem_times):
        """Вывод итоговой статистики"""
        
        total_time = time.time() - self.start_time
        
        print("\n" + "=" * 90)
        print("ИТОГОВАЯ СТАТИСТИКА ЭКСПЕРИМЕНТОВ")
        print("=" * 90)
        
        # Время выполнения
        print(f"\n ВРЕМЯ ВЫПОЛНЕНИЯ:")
        print("-" * 40)
        for problem_name, p_time in problem_times.items():
            title = self.problem_titles.get(problem_name, problem_name)
            print(f"  {title}: {p_time:.2f} с")
        print(f"  {'='*30}")
        print(f"  ВСЕГО: {total_time:.2f} с ({total_time/60:.2f} мин)")
        
        # Лучшие алгоритмы по задачам
        print(f"\n ЛУЧШИЕ АЛГОРИТМЫ ПО ЗАДАЧАМ:")
        print("-" * 40)
        
        for problem_name in self.problems.keys():
            if problem_name in self.results:
                problem_results = self.results[problem_name]
                
                # Находим алгоритм с минимальным средним фитнесом
                best_algo = None
                best_fitness = float('inf')
                
                for algo_name, metrics in problem_results.items():
                    if 'error' not in metrics:
                        if metrics['best_fitness_mean'] < best_fitness:
                            best_fitness = metrics['best_fitness_mean']
                            best_algo = algo_name
                
                if best_algo:
                    title = self.problem_titles.get(problem_name, problem_name)
                    metrics = problem_results[best_algo]
                    
                    if best_fitness < 1e6:
                        status = "✅"
                    else:
                        status = "⚠️"
                    
                    print(f"  {status} {title}: {best_algo} "
                          f"(фитнес={best_fitness:.4e}, "
                          f"время={metrics['execution_time_mean']:.3f} с)")
        
        print("\n" + "=" * 90)
        print(" ЭКСПЕРИМЕНТЫ УСПЕШНО ЗАВЕРШЕНЫ")
        print(f" Все результаты в папке: {self.output_dir}")
        print("=" * 90)
    
    def run_visualization(self):
        """Автоматический запуск визуализации"""
        
        print("\n" + "=" * 60)
        print(" АВТОМАТИЧЕСКИЙ ЗАПУСК ВИЗУАЛИЗАЦИИ")
        print("=" * 60)
        
        # Проверяем наличие файлов визуализации
        viz_files = ['visualization.py', 'plot_step_responses.py']
        
        for viz_file in viz_files:
            if os.path.exists(viz_file):
                print(f"\n  Запуск {viz_file}...")
                try:
                    # Запускаем скрипт визуализации
                    result = subprocess.run([sys.executable, viz_file], 
                                          capture_output=True, text=True)
                    
                    if result.returncode == 0:
                        print(f"   {viz_file} выполнен успешно")
                        
                        # Для visualization.py показываем где искать графики
                        if viz_file == 'visualization.py':
                            print(f"     Графики сохранены в папке: plots/")
                        elif viz_file == 'plot_step_responses.py':
                            print(f"     Графики: step_response_comparison.png")
                            print(f"     Метрики: step_response_metrics.csv")
                    else:
                        print(f"   Ошибка в {viz_file}:")
                        print(f"     {result.stderr[:200]}")
                        
                except Exception as e:
                    print(f"   Не удалось запустить {viz_file}: {e}")
            else:
                print(f"   Файл {viz_file} не найден, пропускаем")
        
        print("\n" + "=" * 60)
        print(" ВИЗУАЛИЗАЦИЯ ЗАВЕРШЕНА")
        print("=" * 60)

def main():
    """Основная функция для запуска экспериментов"""
    
    # Параметры эксперимента
    experiment = ComparativeExperiment(
        num_runs=10,      # Количество запусков
        max_iter=100,     # Количество итераций
        pop_size=30,      # Размер популяции
        output_dir="experiment_results",
        auto_visualize=True  # Автоматически запускать визуализацию
    )
    
    # Запускаем все эксперименты
    results = experiment.run_all_experiments()
    
    print("\n" + "=" * 90)
    print(" РАБОТА ЗАВЕРШЕНА! Все результаты и графики готовы.")
    print("=" * 90)

if __name__ == "__main__":
    main()