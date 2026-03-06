"""
Модуль для сбора и анализа метрик оптимизации.
"""

import time
import numpy as np
from typing import Dict, List, Tuple, Any
import psutil
import os
import tracemalloc
import gc

class MetricsCollector:
    """Класс для сбора метрик выполнения алгоритмов"""
    
    def __init__(self):
        self.metrics = {}
        self.current_metrics = {}
        
    def start_measurement(self, algorithm_name: str, problem_name: str):
        """Начало измерения метрик"""
        self.current_metrics = {
            'algorithm': algorithm_name,
            'problem': problem_name,
            'start_time': time.time(),
            'start_memory': psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2,  # MB
            'function_calls': 0
        }
        
        # Запуск трассировки памяти Python
        tracemalloc.start()
    
    def add_function_call(self):
        """Учет вызова целевой функции"""
        if 'function_calls' in self.current_metrics:
            self.current_metrics['function_calls'] += 1
    
    def end_measurement(self, 
                       best_solution: np.ndarray,
                       best_fitness: float,
                       convergence_history: List[float]) -> Dict[str, Any]:
        """Окончание измерения и сбор всех метрик"""
        
        # Время выполнения
        execution_time = time.time() - self.current_metrics['start_time']
        
        # Использование памяти
        current_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
        memory_used = current_memory - self.current_metrics['start_memory']
        
        # Трассировка памяти Python
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Погрешность (если известно оптимальное значение)
        # Для реальных задач оптимальное значение часто неизвестно,
        # поэтому используем относительную погрешность от наилучшего найденного
        if len(convergence_history) > 1:
            initial_error = convergence_history[0]
            final_error = convergence_history[-1]
            if initial_error > 0:
                relative_error_reduction = (initial_error - final_error) / initial_error * 100
            else:
                relative_error_reduction = 0
        else:
            relative_error_reduction = 0
        
        # Скорость сходимости
        if len(convergence_history) > 10:
            # Вычисляем среднюю скорость улучшения за последние 10 итераций
            last_10 = convergence_history[-10:]
            convergence_rate = 0
            for i in range(1, len(last_10)):
                if last_10[i-1] > 0:
                    rate = (last_10[i-1] - last_10[i]) / last_10[i-1]
                    convergence_rate += rate
            convergence_rate = convergence_rate / 9 if len(last_10) > 1 else 0
        else:
            convergence_rate = 0
        
        # Собираем все метрики
        metrics_result = {
            'algorithm': self.current_metrics['algorithm'],
            'problem': self.current_metrics['problem'],
            'best_solution': best_solution.tolist(),
            'best_fitness': float(best_fitness),
            'execution_time': execution_time,
            'function_calls': self.current_metrics['function_calls'],
            'memory_used_mb': memory_used,
            'python_memory_current_kb': current / 1024,
            'python_memory_peak_kb': peak / 1024,
            'convergence_history': convergence_history,
            'iterations': len(convergence_history),
            'relative_error_reduction_%': relative_error_reduction,
            'convergence_rate': convergence_rate,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Очистка
        self.current_metrics = {}
        gc.collect()
        
        return metrics_result
    
    def calculate_statistics(self, all_metrics: List[Dict]) -> Dict[str, Any]:
        """Вычисление статистических показателей по множеству запусков"""
        
        if not all_metrics:
            return {}
        
        # Извлекаем метрики из всех запусков
        best_fitnesses = [m['best_fitness'] for m in all_metrics]
        execution_times = [m['execution_time'] for m in all_metrics]
        function_calls = [m['function_calls'] for m in all_metrics]
        memory_used = [m['memory_used_mb'] for m in all_metrics]
        iterations = [m['iterations'] for m in all_metrics]
        
        # Вычисляем статистики
        stats = {
            'best_fitness': {
                'mean': float(np.mean(best_fitnesses)),
                'std': float(np.std(best_fitnesses)),
                'min': float(np.min(best_fitnesses)),
                'max': float(np.max(best_fitnesses)),
                'median': float(np.median(best_fitnesses))
            },
            'execution_time': {
                'mean': float(np.mean(execution_times)),
                'std': float(np.std(execution_times)),
                'min': float(np.min(execution_times)),
                'max': float(np.max(execution_times))
            },
            'function_calls': {
                'mean': float(np.mean(function_calls)),
                'std': float(np.std(function_calls)),
                'min': int(np.min(function_calls)),
                'max': int(np.max(function_calls))
            },
            'memory_used_mb': {
                'mean': float(np.mean(memory_used)),
                'std': float(np.std(memory_used)),
                'max': float(np.max(memory_used))
            },
            'iterations': {
                'mean': float(np.mean(iterations)),
                'std': float(np.std(iterations))
            },
            'num_runs': len(all_metrics),
            'success_rate': self.calculate_success_rate(all_metrics)
        }
        
        return stats
    
    def calculate_success_rate(self, all_metrics: List[Dict], threshold: float = 1e-6) -> float:
        """Вычисление процента успешных запусков"""
        
        if not all_metrics:
            return 0.0
        
        # Определяем успешные запуски (достигли заданной точности)
        successful = 0
        for metrics in all_metrics:
            if metrics['best_fitness'] <= threshold:
                successful += 1
        
        return (successful / len(all_metrics)) * 100
    
    def generate_report(self, algorithm_name: str, problem_name: str, 
                       all_metrics: List[Dict]) -> str:
        """Генерация текстового отчета"""
        
        stats = self.calculate_statistics(all_metrics)
        
        report = f"""
{'='*80}
ОТЧЕТ ПО АЛГОРИТМУ: {algorithm_name}
ЗАДАЧА: {problem_name}
КОЛИЧЕСТВО ЗАПУСКОВ: {stats['num_runs']}
{'='*80}

СТАТИСТИЧЕСКИЕ ПОКАЗАТЕЛИ:
--------------------
Качество решения (целевая функция):
  Среднее значение: {stats['best_fitness']['mean']:.6e}
  Стандартное отклонение: {stats['best_fitness']['std']:.6e}
  Минимальное значение: {stats['best_fitness']['min']:.6e}
  Максимальное значение: {stats['best_fitness']['max']:.6e}
  Медиана: {stats['best_fitness']['median']:.6e}

Время выполнения:
  Среднее время: {stats['execution_time']['mean']:.4f} сек
  Стандартное отклонение: {stats['execution_time']['std']:.4f} сек
  Минимальное время: {stats['execution_time']['min']:.4f} сек
  Максимальное время: {stats['execution_time']['max']:.4f} сек

Вычислительные затраты:
  Среднее количество вычислений: {stats['function_calls']['mean']:.0f}
  Минимальное количество вычислений: {stats['function_calls']['min']}
  Максимальное количество вычислений: {stats['function_calls']['max']}

Использование памяти:
  Среднее использование: {stats['memory_used_mb']['mean']:.2f} МБ
  Максимальное использование: {stats['memory_used_mb']['max']:.2f} МБ

Скорость сходимости:
  Среднее количество итераций: {stats['iterations']['mean']:.1f}
  Процент успешных запусков: {stats['success_rate']:.1f}%

ЗАКЛЮЧЕНИЕ:
-----------
"""
        
        # Анализ эффективности
        if stats['best_fitness']['mean'] < 1e-3:
            report += "✓ Алгоритм демонстрирует высокую точность решения.\n"
        elif stats['best_fitness']['mean'] < 1e-1:
            report += "○ Алгоритм показывает удовлетворительную точность.\n"
        else:
            report += "✗ Точность алгоритма недостаточна для данной задачи.\n"
        
        if stats['execution_time']['mean'] < 1.0:
            report += "✓ Алгоритм работает быстро.\n"
        elif stats['execution_time']['mean'] < 5.0:
            report += "○ Скорость работы алгоритма приемлема.\n"
        else:
            report += "✗ Алгоритм работает медленно.\n"
        
        if stats['memory_used_mb']['mean'] < 10:
            report += "✓ Алгоритм экономичен по памяти.\n"
        elif stats['memory_used_mb']['mean'] < 50:
            report += "○ Использование памяти умеренное.\n"
        else:
            report += "✗ Алгоритм потребляет много памяти.\n"
        
        report += f"\n{'='*80}\n"
        
        return report