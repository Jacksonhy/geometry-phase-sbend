"""Assemble complete report and hash all new artifacts after verification."""
import runpy
from common import *
def main():
    assert read(OUT/'verification.json')['passed']
    assert read(OUT/'statistics_verification.json')['passed']
    verify_freeze()
    for name,h in read(OUT/'reporting_sources_before_outcomes.json')['sources'].items():assert sha(OUT/name)==h
    runpy.run_path(str(OUT/'report.py'),run_name='__main__')
    path=OUT/'S-bend正式五组实验报告.md'
    text=path.read_text(encoding='utf-8')
    text+='\n\n## 8. 效应大小与负例全文\n\n'+(OUT/'正式结果负例与效应大小补充.md').read_text(encoding='utf-8')
    text+='\n\n## 9. 分端口指标全文\n\n'+(OUT/'逐端口代理指标.md').read_text(encoding='utf-8')
    text+='\n\n## 10. 图、统计复核及交付索引\n\n'
    for name in ['formal_five_methods.png','paired_geometry_scatter.png','direction_and_region_results.png']:
        text+='!['+name+'](D:/game/research_sbend_formal/'+name+')\n\n'
    text+='主要best/last配对均差、改善数及bootstrap区间还由独立脚本从保存CSV复算通过，见[统计复核](D:/game/research_sbend_formal/statistics_verification.json)。\n\n'
    text+='核验初版逐例重复解压预测数组，运行中为避免冗余I/O停止；原脚本和结果前哈希保留。缓存读取版verify_results_cached_r1.py仅改变读取位置，数值断言不变，已完整完成；版本记录见verification_io_revision.json。正式evaluate.py、权重、预测和主指标从未因此修改或重跑。\n\n'
    text+='输入范围为预定8档宽度、固定二维网格/波长的同分布几何测试，不是多波长、三维、工艺扰动或OOD认证。初始模型种子仅2026；微小强基线差异不应扩张为训练随机性结论。\n\n'
    text+='推理计时来自同一次正式评价中的GPU同步区间，包含场恢复；首模型含初始化后的冷启动影响，未做独立热身/多次性能基准。五组主干相同，不能将秒级差异解释成架构本征速度差；没有为计时重新推理测试集。\n\n'
    text+='[完整输出SHA-256清单](D:/game/research_sbend_formal/outputs_manifest.json)保存本目录交付文件哈希。未改历史实验，不再进行测试后算法开发。\n'
    path.write_text(text,encoding='utf-8')
    inventory={str(p.relative_to(OUT)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in OUT.rglob('*')
        if p.is_file() and p.name!='outputs_manifest.json' and '__pycache__' not in str(p)}
    write('outputs_manifest.json',{'time_utc':now(),'files':inventory,'file_count':len(inventory),
        'total_bytes':sum(r['bytes'] for r in inventory.values()),'formal_evaluation_events':1})
    print('COMPLETE CHINESE REPORT AND ARTIFACT HASHES WRITTEN',flush=True)
if __name__=='__main__':main()
