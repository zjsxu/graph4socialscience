#!/usr/bin/env python3
"""
性能基准测试运行脚本

执行完整的性能基准测试并生成综合报告。
"""

import json
import time
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any

def run_performance_tests():
    """运行性能测试"""
    print("开始执行性能基准测试...")
    
    # 运行各类性能测试
    test_classes = [
        "TestProcessingPerformance",
        "TestMemoryPerformance", 
        "TestConcurrencyPerformance",
        "TestScalabilityPerformance"
    ]
    
    results = {}
    
    for test_class in test_classes:
        print(f"\n运行 {test_class}...")
        
        try:
            # 运行测试类
            cmd = [
                sys.executable, "-m", "pytest", 
                f"tests/test_performance_benchmarks.py::{test_class}",
                "-v", "-s", "--tb=short"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✓ {test_class} 测试通过")
                results[test_class] = {
                    'status': 'passed',
                    'output': result.stdout
                }
            else:
                print(f"✗ {test_class} 测试失败")
                results[test_class] = {
                    'status': 'failed',
                    'output': result.stdout,
                    'error': result.stderr
                }
                
        except Exception as e:
            print(f"✗ {test_class} 执行异常: {e}")
            results[test_class] = {
                'status': 'error',
                'error': str(e)
            }
    
    return results

def generate_comprehensive_report(test_results: Dict[str, Any]) -> Dict[str, Any]:
    """生成综合性能报告"""
    
    # 系统信息
    try:
        import psutil
        system_info = {
            'cpu_count': psutil.cpu_count(),
            'cpu_freq': psutil.cpu_freq().current if psutil.cpu_freq() else 'Unknown',
            'memory_total': psutil.virtual_memory().total // (1024**3),  # GB
            'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            'platform': sys.platform
        }
    except ImportError:
        system_info = {
            'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            'platform': sys.platform
        }
    
    # 分析测试结果
    passed_tests = sum(1 for r in test_results.values() if r['status'] == 'passed')
    failed_tests = sum(1 for r in test_results.values() if r['status'] == 'failed')
    error_tests = sum(1 for r in test_results.values() if r['status'] == 'error')
    
    # 提取性能指标
    performance_metrics = {}
    
    for test_class, result in test_results.items():
        if result['status'] == 'passed' and 'output' in result:
            output = result['output']
            
            # 提取关键性能数据
            if 'docs/sec' in output:
                # 文本处理速度
                lines = output.split('\n')
                for line in lines:
                    if 'docs/sec' in line and 'Speed:' in line:
                        try:
                            speed = float(line.split('Speed:')[1].split('docs/sec')[0].strip())
                            performance_metrics[f'{test_class}_processing_speed'] = speed
                        except:
                            pass
            
            if 'Memory Usage Scaling:' in output:
                # 内存使用情况
                performance_metrics[f'{test_class}_memory_test'] = 'completed'
            
            if 'Speedup:' in output:
                # 并行处理加速比
                lines = output.split('\n')
                for line in lines:
                    if 'Speedup:' in line:
                        try:
                            speedup = float(line.split('Speedup:')[1].split('x')[0].strip())
                            performance_metrics[f'{test_class}_speedup'] = speedup
                        except:
                            pass
    
    # 生成报告
    report = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'system_info': system_info,
        'test_summary': {
            'total_test_classes': len(test_results),
            'passed': passed_tests,
            'failed': failed_tests,
            'errors': error_tests,
            'success_rate': passed_tests / len(test_results) if test_results else 0
        },
        'performance_metrics': performance_metrics,
        'test_details': test_results,
        'recommendations': generate_recommendations(test_results, performance_metrics)
    }
    
    return report

def generate_recommendations(test_results: Dict[str, Any], metrics: Dict[str, Any]) -> list:
    """生成性能优化建议"""
    recommendations = []
    
    # 基于测试结果生成建议
    failed_count = sum(1 for r in test_results.values() if r['status'] != 'passed')
    
    if failed_count > 0:
        recommendations.append(f"有 {failed_count} 个测试类未通过，建议检查相关功能实现")
    
    # 基于性能指标生成建议
    processing_speeds = [v for k, v in metrics.items() if 'processing_speed' in k]
    if processing_speeds:
        avg_speed = sum(processing_speeds) / len(processing_speeds)
        if avg_speed < 100:
            recommendations.append("文档处理速度较慢，建议优化文本处理算法或启用并行处理")
        elif avg_speed > 1000:
            recommendations.append("文档处理速度良好，系统性能表现优秀")
    
    speedups = [v for k, v in metrics.items() if 'speedup' in k]
    if speedups:
        avg_speedup = sum(speedups) / len(speedups)
        if avg_speedup < 1.5:
            recommendations.append("并行处理效果有限，建议检查任务分解策略或增加CPU核心数")
        elif avg_speedup > 2.0:
            recommendations.append("并行处理效果良好，系统具备良好的可扩展性")
    
    if not recommendations:
        recommendations.append("系统性能表现正常，建议定期监控以确保持续稳定")
    
    return recommendations

def generate_markdown_report(report: Dict[str, Any]) -> str:
    """生成Markdown格式报告"""
    lines = []
    
    # 标题
    lines.append("# 语义共词网络分析管线 - 性能基准测试报告")
    lines.append("")
    lines.append(f"**生成时间**: {report['timestamp']}")
    lines.append("")
    
    # 系统信息
    lines.append("## 系统环境")
    lines.append("")
    system_info = report['system_info']
    for key, value in system_info.items():
        lines.append(f"- **{key}**: {value}")
    lines.append("")
    
    # 测试摘要
    lines.append("## 测试摘要")
    lines.append("")
    summary = report['test_summary']
    lines.append(f"- **总测试类数**: {summary['total_test_classes']}")
    lines.append(f"- **通过**: {summary['passed']}")
    lines.append(f"- **失败**: {summary['failed']}")
    lines.append(f"- **错误**: {summary['errors']}")
    lines.append(f"- **成功率**: {summary['success_rate']:.1%}")
    lines.append("")
    
    # 性能指标
    if report['performance_metrics']:
        lines.append("## 关键性能指标")
        lines.append("")
        for metric, value in report['performance_metrics'].items():
            if isinstance(value, (int, float)):
                if 'speed' in metric:
                    lines.append(f"- **{metric}**: {value:.2f} docs/sec")
                elif 'speedup' in metric:
                    lines.append(f"- **{metric}**: {value:.2f}x")
                else:
                    lines.append(f"- **{metric}**: {value}")
            else:
                lines.append(f"- **{metric}**: {value}")
        lines.append("")
    
    # 测试详情
    lines.append("## 测试详情")
    lines.append("")
    for test_class, result in report['test_details'].items():
        status_icon = "✓" if result['status'] == 'passed' else "✗"
        lines.append(f"### {status_icon} {test_class}")
        lines.append("")
        lines.append(f"**状态**: {result['status']}")
        lines.append("")
        
        if result['status'] == 'passed' and 'output' in result:
            # 提取关键输出信息
            output_lines = result['output'].split('\n')
            benchmark_lines = [line for line in output_lines if any(keyword in line for keyword in 
                ['Benchmark:', 'Speed:', 'Processing time:', 'Speedup:', 'Memory Usage:', 'docs/sec'])]
            
            if benchmark_lines:
                lines.append("**关键指标**:")
                lines.append("```")
                for line in benchmark_lines[-10:]:  # 最后10行关键信息
                    if line.strip():
                        lines.append(line.strip())
                lines.append("```")
        elif result['status'] != 'passed':
            if 'error' in result:
                lines.append("**错误信息**:")
                lines.append("```")
                lines.append(result['error'][:500] + "..." if len(result['error']) > 500 else result['error'])
                lines.append("```")
        
        lines.append("")
    
    # 优化建议
    lines.append("## 优化建议")
    lines.append("")
    for rec in report['recommendations']:
        lines.append(f"- {rec}")
    lines.append("")
    
    # 结论
    lines.append("## 结论")
    lines.append("")
    success_rate = report['test_summary']['success_rate']
    if success_rate >= 0.9:
        grade = "A"
        conclusion = "系统性能表现优秀"
    elif success_rate >= 0.7:
        grade = "B"
        conclusion = "系统性能表现良好"
    elif success_rate >= 0.5:
        grade = "C"
        conclusion = "系统性能需要改进"
    else:
        grade = "D"
        conclusion = "系统性能存在严重问题"
    
    lines.append(f"**整体评级**: {grade}")
    lines.append("")
    lines.append(f"{conclusion}，建议根据上述优化建议进行相应调整以提升系统性能。")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*此报告由语义共词网络分析管线自动生成*")
    
    return "\n".join(lines)

def main():
    """主函数"""
    print("语义共词网络分析管线 - 性能基准测试")
    print("=" * 50)
    
    # 运行性能测试
    test_results = run_performance_tests()
    
    # 生成综合报告
    print("\n生成性能报告...")
    report = generate_comprehensive_report(test_results)
    
    # 保存报告
    output_dir = Path("output/performance")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # JSON报告
    json_file = output_dir / f"comprehensive_performance_report_{int(time.time())}.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # Markdown报告
    markdown_content = generate_markdown_report(report)
    markdown_file = output_dir / f"comprehensive_performance_report_{int(time.time())}.md"
    with open(markdown_file, 'w', encoding='utf-8') as f:
        f.write(markdown_content)
    
    print(f"\n性能测试完成！")
    print(f"JSON报告: {json_file}")
    print(f"Markdown报告: {markdown_file}")
    
    # 显示摘要
    print(f"\n测试摘要:")
    print(f"- 总测试类: {report['test_summary']['total_test_classes']}")
    print(f"- 通过: {report['test_summary']['passed']}")
    print(f"- 失败: {report['test_summary']['failed']}")
    print(f"- 成功率: {report['test_summary']['success_rate']:.1%}")
    
    return 0 if report['test_summary']['success_rate'] >= 0.7 else 1

if __name__ == "__main__":
    sys.exit(main())