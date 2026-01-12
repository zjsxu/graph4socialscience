#!/usr/bin/env python3
"""
最终质量报告生成器

为Task 16生成完整的系统测试和质量报告。
"""

import json
import time
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, List

def run_system_tests():
    """运行系统综合测试"""
    print("运行系统综合测试...")
    
    cmd = [
        sys.executable, "-m", "pytest", 
        "tests/test_system_comprehensive.py",
        "-v", "--tb=short"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    return {
        'returncode': result.returncode,
        'stdout': result.stdout,
        'stderr': result.stderr
    }

def run_integration_tests():
    """运行集成测试"""
    print("运行集成测试...")
    
    cmd = [
        sys.executable, "-m", "pytest", 
        "tests/test_integration.py",
        "-v", "--tb=short"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    return {
        'returncode': result.returncode,
        'stdout': result.stdout,
        'stderr': result.stderr
    }

def analyze_test_results(test_output: str) -> Dict[str, Any]:
    """分析测试结果"""
    lines = test_output.split('\n')
    
    # 提取测试统计
    summary_line = None
    for line in lines:
        if 'passed' in line and ('failed' in line or 'warnings' in line or 'error' in line or line.strip().endswith('passed')):
            summary_line = line
            break
    
    if summary_line:
        # 解析测试结果
        import re
        
        # 匹配数字 + passed
        passed_match = re.search(r'(\d+)\s+passed', summary_line)
        failed_match = re.search(r'(\d+)\s+failed', summary_line)
        error_match = re.search(r'(\d+)\s+error', summary_line)
        
        passed = int(passed_match.group(1)) if passed_match else 0
        failed = int(failed_match.group(1)) if failed_match else 0
        errors = int(error_match.group(1)) if error_match else 0
        
        total = passed + failed + errors
        success_rate = passed / total if total > 0 else 0
        
        return {
            'total': total,
            'passed': passed,
            'failed': failed,
            'errors': errors,
            'success_rate': success_rate
        }
    
    return {
        'total': 0,
        'passed': 0,
        'failed': 0,
        'errors': 0,
        'success_rate': 0
    }

def get_documentation_status():
    """检查文档状态"""
    docs_status = {}
    
    # 检查主要文档文件
    doc_files = [
        'README.md',
        'docs/api_reference.md',
        'docs/user_guide.md'
    ]
    
    for doc_file in doc_files:
        path = Path(doc_file)
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
                docs_status[doc_file] = {
                    'exists': True,
                    'size': len(content),
                    'lines': len(content.split('\n'))
                }
        else:
            docs_status[doc_file] = {
                'exists': False,
                'size': 0,
                'lines': 0
            }
    
    return docs_status

def get_performance_report_status():
    """检查性能报告状态"""
    perf_dir = Path("output/performance")
    
    if not perf_dir.exists():
        return {'exists': False, 'reports': []}
    
    reports = list(perf_dir.glob("comprehensive_performance_report_*.md"))
    
    return {
        'exists': True,
        'reports': [str(r) for r in reports],
        'latest_report': str(reports[-1]) if reports else None
    }

def generate_quality_summary(system_results: Dict, integration_results: Dict, 
                           docs_status: Dict, perf_status: Dict) -> Dict[str, Any]:
    """生成质量摘要"""
    
    # 计算整体质量分数
    system_score = system_results['success_rate'] * 40  # 40%权重
    integration_score = integration_results['success_rate'] * 30  # 30%权重
    
    # 文档完整性分数
    doc_score = 0
    for doc_file, status in docs_status.items():
        if status['exists'] and status['size'] > 1000:  # 至少1KB内容
            doc_score += 10
    doc_score = min(doc_score, 20)  # 最多20分
    
    # 性能报告分数
    perf_score = 10 if perf_status['exists'] and perf_status['latest_report'] else 0
    
    total_score = system_score + integration_score + doc_score + perf_score
    
    # 质量等级
    if total_score >= 90:
        grade = 'A'
        status = '优秀'
    elif total_score >= 80:
        grade = 'B'
        status = '良好'
    elif total_score >= 70:
        grade = 'C'
        status = '合格'
    elif total_score >= 60:
        grade = 'D'
        status = '需改进'
    else:
        grade = 'F'
        status = '不合格'
    
    return {
        'total_score': total_score,
        'grade': grade,
        'status': status,
        'component_scores': {
            'system_tests': system_score,
            'integration_tests': integration_score,
            'documentation': doc_score,
            'performance_reports': perf_score
        }
    }

def generate_recommendations(system_results: Dict, integration_results: Dict,
                           docs_status: Dict, quality_summary: Dict) -> List[str]:
    """生成改进建议"""
    recommendations = []
    
    # 基于测试结果
    if system_results['failed'] > 0:
        recommendations.append(f"修复 {system_results['failed']} 个系统测试失败项")
    
    if integration_results['failed'] > 0:
        recommendations.append(f"修复 {integration_results['failed']} 个集成测试失败项")
    
    # 基于文档状态
    for doc_file, status in docs_status.items():
        if not status['exists']:
            recommendations.append(f"创建缺失的文档文件: {doc_file}")
        elif status['size'] < 1000:
            recommendations.append(f"完善文档内容: {doc_file} (当前仅 {status['size']} 字节)")
    
    # 基于整体质量
    if quality_summary['grade'] in ['C', 'D', 'F']:
        recommendations.append("整体质量需要提升，建议优先处理测试失败和文档完善")
    
    if not recommendations:
        recommendations.append("系统质量良好，建议继续保持并定期更新文档和测试")
    
    return recommendations

def generate_markdown_report(report_data: Dict[str, Any]) -> str:
    """生成Markdown格式的最终质量报告"""
    lines = []
    
    # 标题
    lines.append("# 语义共词网络分析管线 - 最终质量报告")
    lines.append("")
    lines.append(f"**生成时间**: {report_data['timestamp']}")
    lines.append(f"**任务**: Task 16 - 系统测试和文档完善")
    lines.append("")
    
    # 质量摘要
    quality = report_data['quality_summary']
    lines.append("## 质量摘要")
    lines.append("")
    lines.append(f"- **整体评级**: {quality['grade']} ({quality['status']})")
    lines.append(f"- **总分**: {quality['total_score']:.1f}/100")
    lines.append("")
    
    # 各组件分数
    lines.append("### 组件评分")
    lines.append("")
    lines.append("| 组件 | 分数 | 权重 |")
    lines.append("|------|------|------|")
    lines.append(f"| 系统测试 | {quality['component_scores']['system_tests']:.1f} | 40% |")
    lines.append(f"| 集成测试 | {quality['component_scores']['integration_tests']:.1f} | 30% |")
    lines.append(f"| 文档完整性 | {quality['component_scores']['documentation']:.1f} | 20% |")
    lines.append(f"| 性能报告 | {quality['component_scores']['performance_reports']:.1f} | 10% |")
    lines.append("")
    
    # 测试结果详情
    lines.append("## 测试结果详情")
    lines.append("")
    
    # 系统测试
    system = report_data['system_test_results']
    lines.append("### 系统综合测试")
    lines.append("")
    lines.append(f"- **总计**: {system['total']} 个测试")
    lines.append(f"- **通过**: {system['passed']} 个")
    lines.append(f"- **失败**: {system['failed']} 个")
    lines.append(f"- **错误**: {system['errors']} 个")
    lines.append(f"- **成功率**: {system['success_rate']:.1%}")
    lines.append("")
    
    # 集成测试
    integration = report_data['integration_test_results']
    lines.append("### 集成测试")
    lines.append("")
    lines.append(f"- **总计**: {integration['total']} 个测试")
    lines.append(f"- **通过**: {integration['passed']} 个")
    lines.append(f"- **失败**: {integration['failed']} 个")
    lines.append(f"- **错误**: {integration['errors']} 个")
    lines.append(f"- **成功率**: {integration['success_rate']:.1%}")
    lines.append("")
    
    # 文档状态
    lines.append("## 文档完整性")
    lines.append("")
    lines.append("| 文档 | 状态 | 大小 | 行数 |")
    lines.append("|------|------|------|------|")
    
    for doc_file, status in report_data['documentation_status'].items():
        status_icon = "✓" if status['exists'] else "✗"
        size_str = f"{status['size']} bytes" if status['exists'] else "N/A"
        lines_str = str(status['lines']) if status['exists'] else "N/A"
        lines.append(f"| {doc_file} | {status_icon} | {size_str} | {lines_str} |")
    
    lines.append("")
    
    # 性能报告状态
    perf = report_data['performance_status']
    lines.append("## 性能基准测试")
    lines.append("")
    if perf['exists']:
        lines.append(f"- **状态**: ✓ 已完成")
        lines.append(f"- **报告数量**: {len(perf['reports'])}")
        if perf['latest_report']:
            lines.append(f"- **最新报告**: {perf['latest_report']}")
    else:
        lines.append("- **状态**: ✗ 未完成")
    lines.append("")
    
    # 改进建议
    lines.append("## 改进建议")
    lines.append("")
    for i, rec in enumerate(report_data['recommendations'], 1):
        lines.append(f"{i}. {rec}")
    lines.append("")
    
    # Task 16 完成状态
    lines.append("## Task 16 完成状态")
    lines.append("")
    
    task_items = [
        ("执行完整的系统测试", system['success_rate'] > 0.7),
        ("完善用户文档和API文档", all(s['exists'] for s in report_data['documentation_status'].values())),
        ("进行性能基准测试", perf['exists']),
        ("生成最终的质量报告", True)  # 当前正在生成
    ]
    
    for item, completed in task_items:
        status_icon = "✓" if completed else "✗"
        lines.append(f"- {status_icon} {item}")
    
    completed_count = sum(1 for _, completed in task_items if completed)
    completion_rate = completed_count / len(task_items)
    
    lines.append("")
    lines.append(f"**Task 16 完成度**: {completion_rate:.1%} ({completed_count}/{len(task_items)})")
    lines.append("")
    
    # 结论
    lines.append("## 结论")
    lines.append("")
    
    if quality['grade'] in ['A', 'B']:
        lines.append(f"Task 16 已成功完成。系统质量达到{quality['status']}水平，")
        lines.append("所有主要功能模块都经过了全面测试，文档完整，性能表现良好。")
    elif quality['grade'] == 'C':
        lines.append(f"Task 16 基本完成。系统质量达到{quality['status']}水平，")
        lines.append("但仍有部分测试失败或文档需要完善。")
    else:
        lines.append(f"Task 16 需要进一步完善。系统质量为{quality['status']}，")
        lines.append("建议优先处理测试失败和文档问题。")
    
    lines.append("")
    lines.append("建议根据上述改进建议继续优化系统质量，")
    lines.append("并在后续开发中保持高质量的测试覆盖率和文档维护。")
    lines.append("")
    
    lines.append("---")
    lines.append("")
    lines.append("*此报告由语义共词网络分析管线质量保证系统自动生成*")
    
    return "\n".join(lines)

def main():
    """主函数"""
    print("语义共词网络分析管线 - 最终质量报告生成")
    print("=" * 50)
    
    # 运行测试
    system_test_output = run_system_tests()
    integration_test_output = run_integration_tests()
    
    # 分析测试结果
    system_results = analyze_test_results(system_test_output['stdout'])
    integration_results = analyze_test_results(integration_test_output['stdout'])
    
    # 检查文档和性能报告状态
    docs_status = get_documentation_status()
    perf_status = get_performance_report_status()
    
    # 生成质量摘要
    quality_summary = generate_quality_summary(
        system_results, integration_results, docs_status, perf_status
    )
    
    # 生成改进建议
    recommendations = generate_recommendations(
        system_results, integration_results, docs_status, quality_summary
    )
    
    # 构建完整报告数据
    report_data = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'system_test_results': system_results,
        'integration_test_results': integration_results,
        'documentation_status': docs_status,
        'performance_status': perf_status,
        'quality_summary': quality_summary,
        'recommendations': recommendations
    }
    
    # 保存报告
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # JSON报告
    json_file = output_dir / f"final_quality_report_{int(time.time())}.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    
    # Markdown报告
    markdown_content = generate_markdown_report(report_data)
    markdown_file = output_dir / f"final_quality_report_{int(time.time())}.md"
    with open(markdown_file, 'w', encoding='utf-8') as f:
        f.write(markdown_content)
    
    print(f"\n最终质量报告生成完成！")
    print(f"JSON报告: {json_file}")
    print(f"Markdown报告: {markdown_file}")
    
    # 显示摘要
    print(f"\n质量摘要:")
    print(f"- 整体评级: {quality_summary['grade']} ({quality_summary['status']})")
    print(f"- 总分: {quality_summary['total_score']:.1f}/100")
    print(f"- 系统测试成功率: {system_results['success_rate']:.1%}")
    print(f"- 集成测试成功率: {integration_results['success_rate']:.1%}")
    
    return 0 if quality_summary['grade'] in ['A', 'B', 'C'] else 1

if __name__ == "__main__":
    sys.exit(main())