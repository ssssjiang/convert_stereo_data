#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用 ReportLab 生成专业中文简历 PDF
安装: pip install reportlab
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import os

def register_chinese_fonts():
    """注册中文字体"""
    font_candidates = [
        ("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc", "NotoSans"),
        ("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc", "WenQuanYi"),
        ("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf", "DroidSans"),
        ("/System/Library/Fonts/PingFang.ttc", "PingFang"),  # macOS
    ]
    
    for font_path, font_name in font_candidates:
        if os.path.exists(font_path):
            try:
                pdfmetrics.registerFont(TTFont(font_name, font_path))
                return font_name
            except Exception as e:
                print(f"注册字体 {font_name} 失败: {e}")
                continue
    
    # 如果都没找到，使用默认字体（可能不支持中文）
    print("警告: 未找到合适的中文字体，将使用默认字体")
    return "Helvetica"

def create_styles(font_name="Helvetica"):
    """创建样式"""
    styles = getSampleStyleSheet()
    
    # 标题样式
    styles.add(ParagraphStyle(
        name='ChineseTitle',
        parent=styles['Title'],
        fontName=font_name,
        fontSize=18,
        spaceAfter=12,
        alignment=TA_CENTER,
        textColor=colors.black
    ))
    
    # 联系信息样式
    styles.add(ParagraphStyle(
        name='Contact',
        parent=styles['Normal'],
        fontName=font_name,
        fontSize=11,
        spaceAfter=16,
        alignment=TA_CENTER,
        textColor=colors.grey
    ))
    
    # 章节标题样式
    styles.add(ParagraphStyle(
        name='SectionTitle',
        parent=styles['Heading2'],
        fontName=font_name,
        fontSize=14,
        spaceBefore=16,
        spaceAfter=8,
        textColor=colors.royalblue,
        borderWidth=1,
        borderColor=colors.royalblue,
        borderPadding=5,
        leftIndent=0,
        borderRadius=2
    ))
    
    # 正文样式
    styles.add(ParagraphStyle(
        name='ChineseBody',
        parent=styles['Normal'],
        fontName=font_name,
        fontSize=10,
        leading=14,
        spaceAfter=8,
        alignment=TA_JUSTIFY,
        leftIndent=10,
        rightIndent=10
    ))
    
    # 公司/职位样式
    styles.add(ParagraphStyle(
        name='CompanyTitle',
        parent=styles['Normal'],
        fontName=font_name,
        fontSize=12,
        spaceBefore=12,
        spaceAfter=6,
        textColor=colors.darkblue,
        leftIndent=5
    ))
    
    return styles

def create_resume_pdf():
    """生成简历PDF"""
    # 输出路径
    output_path = "/mnt/data/宋姝-SLAM-简历-ReportLab版.pdf"
    
    # 注册中文字体
    font_name = register_chinese_fonts()
    
    # 创建文档
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        topMargin=2*cm,
        bottomMargin=2*cm,
        leftMargin=2*cm,
        rightMargin=2*cm
    )
    
    # 获取样式
    styles = create_styles(font_name)
    
    # 构建文档内容
    story = []
    
    # 标题和联系信息
    story.append(Paragraph("宋姝 | SLAM算法工程师", styles['ChineseTitle']))
    story.append(Paragraph("手机: 17801020905 | 邮箱: songshu0905@gmail.com", styles['Contact']))
    
    # 简历内容
    resume_sections = [
        ("石头科技 | 算法工程师  2024.XX – 至今", """
<b>项目：室外割草机双目 vSLAM 研发与多传感器融合优化</b><br/>
<br/>
<b>算法研发与架构优化</b><br/>
• 参与 vSLAM 框架重构并完成上机部署，架构由同步流程演进为异步多线程结构，运行效率提升 30%<br/>
• 实现 IMU + 轮速多传感器融合，显著提升定位精度与系统鲁棒性<br/>
<br/>
<b>工具链与数据验证</b><br/>
• 自主开发多传感器数据分析 Toolkit，支持 RTK、双目、IMU、轮速的验证、预处理与可视化分析<br/>
• 工具已在多轮实地测试中稳定应用，显著缩短问题排查周期<br/>
<br/>
<b>标定与算法优化支持</b><br/>
• 参与双目 + IMU 标定流程设计与优化，开发外参验证与误差评估工具<br/>
• 优化标定算法，提高标定精度与一致性<br/>
<br/>
<b>成果亮点</b><br/>
• 掌握双目 vSLAM 全链路（视觉前端、非线性优化、多传感器融合、多线程架构）<br/>
• 推动算法、工程与验证一体化，确保算法模块可落地部署<br/>
• 在标定、跨团队调试及系统验证中提供关键技术支持
        """),
        
        ("XREAL | SLAM算法工程师  2022.11 – 2024.XX", """
<b>项目：Room-scale 视觉惯性建图与定位（Spatial Anchor）</b><br/>
<br/>
• 从0到1研发跨session/跨user地图共享与定位功能，支持Linux/Android多平台API<br/>
• 双目建图与定位：stereo matcher、stereo-IMU联合优化、GPnP双目位姿求解<br/>
• 多种2D-3D匹配策略（active search、两阶段匹配、PQ量化），回环召回率提升20%<br/>
• 优化 lo-ransac + 位姿先验，提升定位精度；IMU预积分模块减少25%优化时间<br/>
• 多地图管理（互助定位/合并/更新）、地图质量评估、跨平台上线<br/>
<br/>
<b>成果：</b>集成至XREAL SDK并于CES展会展示虚拟多屏交互
        """),
        
        ("Ninebot-Segway | SLAM算法工程师  2019.XX – 2022.XX", """
<b>项目：City-scale 大规模室外建图</b><br/>
• 实现 hypermap pipeline（关键帧筛选、odom+global SfM恢复尺度、错误匹配过滤等）<br/>
• 分块-合并建图提升大规模场景下效率与鲁棒性；融合RTK数据约束，精度从米级提升至分米级<br/>
• 搭建benchmark评估传统与learning特征性能<br/>
<b>成果：</b>ICRA论文发表，建图时间缩短至Colmap的1/5<br/>
<br/>
<b>项目：餐厅机器人视觉定位</b><br/>
• 将tag作为6DoF landmark融入视觉建图框架，pnp定位结果接入融合模块<br/>
• 设计红外摄像头标定SOP并交付工厂操作<br/>
<br/>
<b>项目：融合odom优化VIO位姿估计</b><br/>
• 实现IMU加速度计bias初始化；引入odom观测约束解决不可观问题
        """),
        
        ("竞赛与荣誉", """
• IROS Lifelong SLAM Challenge 第三名
        """),
        
        ("技能", """
<b>编程语言：</b>C++（精通）、Python（熟练）<br/>
<b>算法框架：</b>vins-mono, maplab, Colmap, OpenMVG<br/>
<b>技术领域：</b>多视图几何、非线性优化、双目/IMU融合、回环检测、地图管理<br/>
<b>工具链：</b>Linux, Git, CMake, OpenCV, Eigen
        """)
    ]
    
    # 添加各个章节
    for section_title, section_content in resume_sections:
        story.append(Paragraph(section_title, styles['SectionTitle']))
        story.append(Paragraph(section_content.strip(), styles['ChineseBody']))
        story.append(Spacer(1, 6))
    
    # 构建PDF
    doc.build(story)
    
    print(f"简历已生成: {output_path}")
    return output_path

if __name__ == "__main__":
    create_resume_pdf()
