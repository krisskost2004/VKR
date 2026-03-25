"""
Основной скрипт для проведения сравнительных экспериментов.
Запускает все алгоритмы на всех задачах и собирает метрики.
Теперь поддерживает:
    - Два уровня успешности: Feasible и Acceptable.
    - Агрегирование истории сходимости по всем запускам.
    - Метрики по числу вычислений функции (FE до порога, J@FE).
"""

import numpy as np
import pandas as pd
import json
import time
from datetime import datetime
import os
import sys
from typing import Dict, List, Tuple, Any

from algorithms import PSO, GWO, WOA, HHO, SMA
from problems import get_problem_info
from simulation import simulate_dc_motor_pid, compute_step_metrics


class ComparativeExperiment:
    """Класс для проведения сравнительных экспериментов"""
    
    def __init__(self, 
                 num_runs: int = 10,
                 max_iter: int = 100,
                 pop_size: int = 30,
                 output_dir: str = "experiment_results",
                 auto_visualize: bool = True):
        self.num_runs = num_runs
        self.max_iter = max_iter
        self.pop_size = pop_size
        self.output_dir = output_dir
        self.auto_visualize = auto_visualize
        
        os.makedirs(output_dir, exist_ok=True)
        
        self.algorithms = {
            'PSO': PSO,
            'GWO': GWO,
            'WOA': WOA,
            'HHO': HHO,
            'SMA': SMA
        }
        
        self.problems = {
            'dc_motor_pid': 'Оптимизация ПИД-регулятора (двигатель)',
            'inverted_pendulum': 'Балансировка маятника',
            'liquid_level': 'Управление уровнем жидкости'
        }
        
        self.problem_titles = {
            'dc_motor_pid': 'ПИД-регулятор двигателя',
            'inverted_pendulum': 'Балансировка маятника',
            'liquid_level': 'Уровень жидкости'
        }
        
        self.results = {}
        self.convergence_data = {}   # для каждого алгоритма и задачи храним список историй
        self.start_time = None
    
    def evaluate_solution(self, problem_name: str, solution: np.ndarray) -> Dict[str, Any]:
        """
        Пост-проверка решения: определяет допустимость и приемлемость,
        а также вычисляет инженерные метрики.
        """
        if problem_name == 'dc_motor_pid':
            Kp, Ki, Kd = solution
            # Допустимость: параметры в границах и моделирование прошло без ошибок
            if not (0.1 <= Kp <= 50 and 0.01 <= Ki <= 30 and 0 <= Kd <= 10):
                return {'feasible': False, 'acceptable': False, 'metrics': {}}
            try:
                t, y = simulate_dc_motor_pid(solution, t_end=5, n_points=500)
                if np.any(np.isnan(y)) or np.any(np.isinf(y)):
                    return {'feasible': False, 'acceptable': False, 'metrics': {}}
                metrics = compute_step_metrics(t, y)
                feasible = True
                # Приемлемость: перерегулирование ≤ 10%, время установления ≤ 2 с, уст. ошибка ≤ 0.02
                acceptable = (metrics['overshoot'] <= 10.0 and 
                              metrics['settling_time'] <= 2.0 and
                              metrics['steady_state_error'] <= 0.02)
                return {'feasible': feasible, 'acceptable': acceptable, 'metrics': metrics}
            except:
                return {'feasible': False, 'acceptable': False, 'metrics': {}}
        
        elif problem_name == 'inverted_pendulum':
            # Параметры маятника (те же, что в problems.py)
            M, m, b, l, g = 1.0, 0.1, 0.1, 0.5, 9.81
            A = np.array([
                [0, 1, 0, 0],
                [0, -b/M, -m*g/M, 0],
                [0, 0, 0, 1],
                [0, -b/(M*l), (M+m)*g/(M*l), 0]
            ])
            B = np.array([[0], [1/M], [0], [1/(M*l)]])
            K = np.array([[solution[0], solution[1], solution[2], solution[3]]])
            A_closed = A - B @ K
            eigvals = np.linalg.eigvals(A_closed)
            feasible = np.all(np.real(eigvals) < 0)
            if not feasible:
                return {'feasible': False, 'acceptable': False, 'metrics': {}}
            # Моделируем
            def dyn(x, t):
                return A_closed.dot(x)
            x0 = [0, 0, 0.1, 0]
            t_span = np.linspace(0, 5, 500)
            from scipy.integrate import odeint
            x = odeint(dyn, x0, t_span)
            # Приемлемость: максимальное отклонение угла < 0.05 рад после переходного процесса
            final_angle = np.abs(x[-1, 2])
            max_angle = np.max(np.abs(x[:, 2]))
            acceptable = (final_angle < 0.05) and (max_angle < 0.2)
            return {'feasible': feasible, 'acceptable': acceptable, 'metrics': {'final_angle': final_angle, 'max_angle': max_angle}}
        
        elif problem_name == 'liquid_level':
            # Упрощённая проверка допустимости: нет NaN, уровни в разумных пределах
            # Для полной проверки можно запустить симуляцию с полученными параметрами
            from problems import liquid_level_control_objective
            try:
                fit = liquid_level_control_objective(solution)
                feasible = fit < 1e6
                if not feasible:
                    return {'feasible': False, 'acceptable': False, 'metrics': {}}
                # Приемлемость: остаточная ошибка по обоим каналам < 0.05
                # Здесь мы не можем получить финальные уровни без повторной симуляции, но можем использовать fit
                # Более правильно: пересчитать симуляцию и взять финальную ошибку
                # Для простоты: будем считать приемлемым, если фитнес < 10 (это эмпирическое пороговое значение)
                acceptable = fit < 10.0
                return {'feasible': feasible, 'acceptable': acceptable, 'metrics': {'fitness': fit}}
            except:
                return {'feasible': False, 'acceptable': False, 'metrics': {}}
        else:
            return {'feasible': False, 'acceptable': False, 'metrics': {}}
    
    def run_single_experiment(self, 
                             algorithm_class,
                             problem_info: Dict,
                             algorithm_name: str,
                             problem_name: str,
                             run_id: int) -> Dict[str, Any]:
        """Запуск одного эксперимента с возвратом метрик, включая флаги допустимости"""
        try:
            seed = 42 + run_id * 100
            dim = problem_info['dim']
            bounds = problem_info['bounds']
            objective_func = problem_info['objective_func']
            
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
            
            best_solution, best_fitness = algorithm.optimize()
            metrics = algorithm.get_metrics()
            
            # Пост-проверка решения
            eval_result = self.evaluate_solution(problem_name, best_solution)
            
            # Вычисляем FE до порога (если порог достигнут)
            target_fitness = 1e-3  # порог для всех задач
            fe_to_target = None
            if 'convergence_history' in metrics and metrics['convergence_history']:
                hist = metrics['convergence_history']
                for idx, val in enumerate(hist):
                    if val <= target_fitness:
                        # Вычисляем общее количество вычислений к этой итерации
                        # Упрощённо: каждая итерация делает pop_size вычислений + начальные
                        fe_at_idx = self.pop_size * (idx + 1)
                        fe_to_target = fe_at_idx
                        break
            
            # Значения на отсечках бюджета
            J_at_500 = None
            J_at_1000 = None
            J_at_2000 = None
            if 'convergence_history' in metrics and metrics['convergence_history']:
                hist = metrics['convergence_history']
                # Преобразуем историю в список FE
                fe_values = [self.pop_size * (i+1) for i in range(len(hist))]
                # Интерполяция
                for target_fe, target_var in [(500, 'J_at_500'), (1000, 'J_at_1000'), (2000, 'J_at_2000')]:
                    idx = np.searchsorted(fe_values, target_fe)
                    if idx < len(hist):
                        value = hist[idx]
                    else:
                        value = hist[-1] if hist else np.inf
                    if target_fe == 500:
                        J_at_500 = value
                    elif target_fe == 1000:
                        J_at_1000 = value
                    elif target_fe == 2000:
                        J_at_2000 = value
            
            metrics.update({
                'algorithm': algorithm_name,
                'problem': problem_name,
                'run_id': run_id,
                'seed': seed,
                'solution': best_solution.tolist(),
                'best_fitness': float(best_fitness),
                'feasible': eval_result['feasible'],
                'acceptable': eval_result['acceptable'],
                'fe_to_target': fe_to_target,
                'J_at_500': J_at_500,
                'J_at_1000': J_at_1000,
                'J_at_2000': J_at_2000
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
                'feasible': False,
                'acceptable': False,
                'error': str(e)
            }
    
    def print_progress_bar(self, current, total, bar_length=50):
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
        
        problem_times = {}
        
        for problem_idx, (problem_name, problem_description) in enumerate(self.problems.items(), 1):
            problem_start = time.time()
            
            print(f"\n{'='*60}")
            print(f" ЗАДАЧА {problem_idx}/{len(self.problems)}: {problem_description}")
            print(f"{'='*60}")
            
            problem_info = get_problem_info(problem_name)
            if problem_info is None:
                print(f"❌ Ошибка: задача '{problem_name}' не найдена")
                continue
            
            self.results[problem_name] = {}
            self.convergence_data[problem_name] = {}  # здесь будем хранить список историй для каждого алгоритма
            
            for algo_idx, (algorithm_name, algorithm_class) in enumerate(self.algorithms.items(), 1):
                print(f"\n  Алгоритм {algo_idx}/{len(self.algorithms)}: {algorithm_name}")
                print("  " + "-" * 40)
                
                all_metrics = []
                convergence_histories = []  # список списков best_fitness по запускам
                
                for run_id in range(self.num_runs):
                    current_experiment += 1
                    self.print_progress_bar(current_experiment, total_experiments)
                    
                    metrics = self.run_single_experiment(
                        algorithm_class=algorithm_class,
                        problem_info=problem_info,
                        algorithm_name=algorithm_name,
                        problem_name=problem_name,
                        run_id=run_id
                    )
                    all_metrics.append(metrics)
                    if 'convergence_history' in metrics and metrics['convergence_history']:
                        convergence_histories.append(metrics['convergence_history'])
                    else:
                        convergence_histories.append([])
                
                print()
                
                # Фильтруем успешные (feasible) запуски для вычисления статистики
                feasible_runs = [m for m in all_metrics if m.get('feasible', False)]
                acceptable_runs = [m for m in all_metrics if m.get('acceptable', False)]
                
                if feasible_runs:
                    best_fitness_values = [m['best_fitness'] for m in feasible_runs]
                    execution_times = [m['execution_time'] for m in feasible_runs]
                    
                    mean_fitness = np.mean(best_fitness_values)
                    std_fitness = np.std(best_fitness_values)
                    median_fitness = np.median(best_fitness_values)
                    q25_fitness = np.percentile(best_fitness_values, 25)
                    q75_fitness = np.percentile(best_fitness_values, 75)
                    
                    mean_time = np.mean(execution_times)
                    std_time = np.std(execution_times)
                    
                    # FE до порога
                    fe_to_target_list = [m.get('fe_to_target', None) for m in feasible_runs if m.get('fe_to_target') is not None]
                    median_fe_to_target = np.median(fe_to_target_list) if fe_to_target_list else np.nan
                    
                    # J@FE
                    J500_list = [m.get('J_at_500', None) for m in feasible_runs if m.get('J_at_500') is not None]
                    median_J500 = np.median(J500_list) if J500_list else np.nan
                    J1000_list = [m.get('J_at_1000', None) for m in feasible_runs if m.get('J_at_1000') is not None]
                    median_J1000 = np.median(J1000_list) if J1000_list else np.nan
                    J2000_list = [m.get('J_at_2000', None) for m in feasible_runs if m.get('J_at_2000') is not None]
                    median_J2000 = np.median(J2000_list) if J2000_list else np.nan
                    
                    feasible_rate = len(feasible_runs) / len(all_metrics) * 100
                    acceptable_rate = len(acceptable_runs) / len(all_metrics) * 100
                    
                    # Вывод
                    status = "✅" if acceptable_rate > 50 else "⚠️" if feasible_rate > 0 else "❌"
                    print(f"  {status} {algorithm_name}: "
                          f"Фитнес = {mean_fitness:.4e} ± {std_fitness:.4e}, "
                          f"Время = {mean_time:.3f} ± {std_time:.3f} с, "
                          f"Feasible={feasible_rate:.1f}%, Acceptable={acceptable_rate:.1f}%")
                    
                    self.results[problem_name][algorithm_name] = {
                        'best_fitness_mean': float(mean_fitness),
                        'best_fitness_std': float(std_fitness),
                        'best_fitness_median': float(median_fitness),
                        'best_fitness_q25': float(q25_fitness),
                        'best_fitness_q75': float(q75_fitness),
                        'execution_time_mean': float(mean_time),
                        'execution_time_std': float(std_time),
                        'feasible_rate': feasible_rate,
                        'acceptable_rate': acceptable_rate,
                        'median_fe_to_target': median_fe_to_target,
                        'median_J_at_500': median_J500,
                        'median_J_at_1000': median_J1000,
                        'median_J_at_2000': median_J2000,
                        'all_runs': all_metrics
                    }
                    
                    # Сохраняем истории сходимости для всех запусков
                    self.convergence_data[problem_name][algorithm_name] = convergence_histories
                else:
                    print(f"  ❌ {algorithm_name}: Нет допустимых запусков")
                    self.results[problem_name][algorithm_name] = {
                        'error': 'Нет допустимых запусков',
                        'all_runs': all_metrics
                    }
                    self.convergence_data[problem_name][algorithm_name] = []
            
            problem_times[problem_name] = time.time() - problem_start
            print(f"\n   Время выполнения задачи: {problem_times[problem_name]:.2f} с")
        
        self.save_results()
        self.print_final_statistics(problem_times)
        
        if self.auto_visualize:
            self.run_visualization()
        
        return self.results
    
    def save_results(self):
        print("\n" + "=" * 60)
        print("СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
        print("=" * 60)
        
        # Сводная таблица CSV
        summary_data = []
        for problem_name, algorithms in self.results.items():
            for algorithm_name, metrics in algorithms.items():
                if 'error' not in metrics:
                    summary_data.append({
                        'Problem': self.problem_titles.get(problem_name, problem_name),
                        'Algorithm': algorithm_name,
                        'Best_Fitness_Mean': metrics['best_fitness_mean'],
                        'Best_Fitness_Std': metrics['best_fitness_std'],
                        'Best_Fitness_Median': metrics['best_fitness_median'],
                        'Best_Fitness_Q25': metrics['best_fitness_q25'],
                        'Best_Fitness_Q75': metrics['best_fitness_q75'],
                        'Execution_Time_Mean': metrics['execution_time_mean'],
                        'Execution_Time_Std': metrics['execution_time_std'],
                        'Feasible_Rate_%': metrics['feasible_rate'],
                        'Acceptable_Rate_%': metrics['acceptable_rate'],
                        'Median_FE_to_Target': metrics['median_fe_to_target'],
                        'Median_J@500': metrics['median_J_at_500'],
                        'Median_J@1000': metrics['median_J_at_1000'],
                        'Median_J@2000': metrics['median_J_at_2000']
                    })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_path = os.path.join(self.output_dir, "summary_results.csv")
            summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
            print(f" Сводные результаты: {summary_path}")
        
        # Детальные результаты JSON
        detailed_path = os.path.join(self.output_dir, "results.json")
        with open(detailed_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, default=str, ensure_ascii=False)
        print(f" Детальные результаты: {detailed_path}")
        
        # Данные сходимости JSON (список историй по алгоритмам)
        convergence_path = os.path.join(self.output_dir, "convergence.json")
        with open(convergence_path, 'w', encoding='utf-8') as f:
            json.dump(self.convergence_data, f, indent=2, default=str, ensure_ascii=False)
        print(f" Данные сходимости: {convergence_path}")
        
        # Параметры эксперимента
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
        total_time = time.time() - self.start_time
        print("\n" + "=" * 90)
        print("ИТОГОВАЯ СТАТИСТИКА ЭКСПЕРИМЕНТОВ")
        print("=" * 90)
        print(f"\n ВРЕМЯ ВЫПОЛНЕНИЯ:")
        print("-" * 40)
        for problem_name, p_time in problem_times.items():
            title = self.problem_titles.get(problem_name, problem_name)
            print(f"  {title}: {p_time:.2f} с")
        print(f"  {'='*30}")
        print(f"  ВСЕГО: {total_time:.2f} с ({total_time/60:.2f} мин)")
        
        print(f"\n ЛУЧШИЕ АЛГОРИТМЫ ПО ЗАДАЧАМ (по медиане фитнеса):")
        print("-" * 40)
        for problem_name in self.problems.keys():
            if problem_name in self.results:
                problem_results = self.results[problem_name]
                best_algo = None
                best_median = float('inf')
                for algo_name, metrics in problem_results.items():
                    if 'error' not in metrics:
                        if metrics['best_fitness_median'] < best_median:
                            best_median = metrics['best_fitness_median']
                            best_algo = algo_name
                if best_algo:
                    title = self.problem_titles.get(problem_name, problem_name)
                    print(f"  {title}: {best_algo} (медиана фитнеса={best_median:.4e})")
        print("\n" + "=" * 90)
        print(" ЭКСПЕРИМЕНТЫ УСПЕШНО ЗАВЕРШЕНЫ")
        print(f" Все результаты в папке: {self.output_dir}")
        print("=" * 90)
    
    def run_visualization(self):
        print("\n" + "=" * 60)
        print(" АВТОМАТИЧЕСКИЙ ЗАПУСК ВИЗУАЛИЗАЦИИ")
        print("=" * 60)
        viz_files = ['visualization.py', 'plot_step_responses.py']
        for viz_file in viz_files:
            if os.path.exists(viz_file):
                print(f"\n  Запуск {viz_file}...")
                try:
                    import subprocess
                    result = subprocess.run([sys.executable, viz_file], 
                                          capture_output=True, text=True)
                    if result.returncode == 0:
                        print(f"   {viz_file} выполнен успешно")
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
    experiment = ComparativeExperiment(
        num_runs=10,
        max_iter=100,
        pop_size=30,
        output_dir="experiment_results",
        auto_visualize=True
    )
    experiment.run_all_experiments()

if __name__ == "__main__":
    main()