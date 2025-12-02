#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用 WeasyPrint 生成现代化中文简历 PDF
安装: pip install weasyprint
"""

import weasyprint
from pathlib import Path

def create_resume_html():
    """创建简历的HTML内容，匹配原版简历格式"""
    html_content = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>宋姝 - SLAM算法工程师</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        @page {
            size: A4;
            margin: 2cm;
        }
        
        body {
            font-family: 'SimSun', '宋体', serif;
            font-size: 12pt;
            line-height: 1.5;
            color: #000;
            background: white;
        }
        
        .header {
            text-align: center;
            margin-bottom: 1.5em;
        }
        
        .name {
            font-size: 18pt;
            font-weight: bold;
            margin-bottom: 0.3em;
        }
        
        .title {
            font-size: 14pt;
            font-weight: bold;
            margin-bottom: 0.5em;
        }
        
        .contact {
            font-size: 11pt;
            margin-bottom: 1em;
        }
        
        .section {
            margin-bottom: 1.2em;
        }
        
        .section-title {
            font-size: 14pt;
            font-weight: bold;
            margin-bottom: 0.5em;
            text-decoration: underline;
        }
        
        .timeline-table {
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 1em;
        }
        
        .timeline-table td {
            vertical-align: top;
            padding: 0.2em 0;
            border: none;
        }
        
        .period-cell {
            width: 25%;
            font-weight: bold;
            padding-right: 1em;
        }
        
        .content-cell {
            width: 75%;
            padding-left: 1em;
        }
        
        .company {
            font-weight: bold;
            margin-bottom: 0.3em;
        }
        
        .project-description {
            margin-bottom: 0.8em;
        }
        
        .project-title {
            font-weight: bold;
            margin-bottom: 0.3em;
        }
        
        .main-work {
            font-weight: bold;
            margin: 0.5em 0 0.3em 0;
        }
        
        ul {
            margin-left: 1em;
            margin-bottom: 0.5em;
        }
        
        li {
            margin-bottom: 0.2em;
            line-height: 1.4;
        }
        
        .achievement {
            margin-top: 0.5em;
            font-weight: bold;
        }
        
        .skills-list {
            margin-left: 0;
        }
        
        .skills-list li {
            margin-bottom: 0.3em;
        }
    </style>
</head>
<body>
    <div class="header">
        <div class="name">宋姝 - SLAM算法工程师</div>
        <div class="contact">
            手机: 17801020905 | 邮箱: songshu0905@gmail.com | 生日: 1997.9
        </div>
    </div>

    <div class="section">
        <div class="section-title">教育背景</div>
        <table class="timeline-table">
            <tr>
                <td class="period-cell">2018.09 - 2021.07</td>
                <td class="content-cell">
                    <div class="company">北京邮电大学</div>
                    <div>信息与通信工程 - 硕士</div>
                </td>
            </tr>
            <tr>
                <td class="period-cell">2014.09 - 2018.07</td>
                <td class="content-cell">
                    <div class="company">北京交通大学</div>
                    <div>信息与通信工程 - 本科</div>
                </td>
            </tr>
        </table>
    </div>

    <div class="section">
        <div class="section-title">工作经历</div>
        <table class="timeline-table">
            <tr>
                <td class="period-cell">XREAL</td>
                <td class="content-cell">
                    <div class="company">SLAM算法工程师</div>
                    <div>2022.11 - 至今</div>
                </td>
            </tr>
            <tr>
                <td class="period-cell">Ninebot-Segway</td>
                <td class="content-cell">
                    <div class="company">SLAM算法工程师</div>
                    <div>2019.11 - 2022.11</div>
                </td>
            </tr>
        </table>
    </div>

    <div class="section">
        <div class="section-title">项目经历</div>
        
        <table class="timeline-table">
            <tr>
                <td class="period-cell">2022.11 - 至今</td>
                <td class="content-cell">
                    <div class="project-title">room-scale 视觉惯性建图与定位</div>
                    <div class="company">XREAL</div>
                    <div class="project-description">
                        <div class="main-work">项目简介:</div>
                        从0到1实现spatial anchor功能，发布API供开发者在多平台（Linux / Android）使用。spatial anchor的功能是对空间环境建图，并实现跨session，跨user定位。
                        <div class="main-work">主要工作:</div>
                        <ul>
                            <li>优化建图和定位效果:</li>
                            <li style="margin-left: 1em;">实现双目建图、定位: stereo matcher、stereo-IMU联合优化、gpnp求解双目pose等;</li>
                            <li style="margin-left: 1em;">实现多种2D - 3D匹配算法，在回环/重定位时，平衡召回率和效率: 包括 active search、两阶段匹配、kd-tree、倒排索引本（online/offline training, multi-index）等匹配算法，以及描述子量化（PCA降维，PQ量化）;</li>
                            <li style="margin-left: 1em;">优化lo-ransac逻辑，结合pose先验提高召回率回率;</li>
                            <li style="margin-left: 1em;">实现IMU预积分模块，减少优化时间;</li>
                            <li>功能扩展:</li>
                            <li style="margin-left: 1em;">实现多地图管理模块，支持地图间互助定位、地图合并与更新;</li>
                            <li style="margin-left: 1em;">设计并导入户地图demo，实现地图质量评断模块;</li>
                            <li>工程实现: 搭建benchmark，设计指标，评估建图定位表现，以及接口定义与实现，交叉编译，移动端上线;</li>
                        </ul>
                        <div class="achievement">项目成果:</div>
                        <ul>
                            <li>spatial anchor SDK文档;</li>
                            <li>CES展会: 在展示小屋中，配合手势与image tracking，用户可以体验在空间中布置虚拟多屏，并进行image tracking切换应用等.</li>
                        </ul>
                    </div>
                </td>
            </tr>
        </table>

        <table class="timeline-table">
            <tr>
                <td class="period-cell">2021.9 - 2022.7</td>
                <td class="content-cell">
                    <div class="project-title">大规模室外场景建图</div>
                    <div class="company">Ninebot-Segway</div>
                    <div class="project-description">
                        <div class="main-work">项目简介:</div>
                        实现大规模室外场景的视觉建图，后续为机器人提供重定位。
                        <div class="main-work">主要工作:</div>
                        <ul>
                            <li>实现hypermap pipeline:</li>
                            <li style="margin-left: 1em;">关键帧筛选: 利用odom做关键帧筛选，减少冗余信息，提高相对平移估计的精度;</li>
                            <li style="margin-left: 1em;">odom + global sfm恢复pose真实尺度: 参考1dsfm等削除相对平移outlier，优化global sfm建图;</li>
                            <li style="margin-left: 1em;">基于global sfm结果进滤错误的双视匹配，为increment sfm提供相对位姿约束，并辅助图像注册等;</li>
                            <li>实现分块-合并建图: 在大规模场景下提升建图效率和鲁棒性;</li>
                            <li>多源数据融合: 添加RTK数据约束，提升建图精度;</li>
                            <li>调研传统learning特征，及特征匹配算法，搭建benchmark，设计指标，在自家 & 开源数据上评估效果;</li>
                        </ul>
                        <div class="achievement">项目成果:</div>
                        <ul>
                            <li>项目主页</li>
                            <li>ICRA2022论文: city-scale建图上，对比colmap，建图精度由来高到分米级，建图时间缩短至1/5;</li>
                        </ul>
                    </div>
                </td>
            </tr>
        </table>

        <table class="timeline-table">
            <tr>
                <td class="period-cell">2023.11 - 2024.1</td>
                <td class="content-cell">
                    <div class="project-title">优化Image Tracking功能</div>
                    <div class="company">XREAL</div>
                    <div class="project-description">
                        <div class="main-work">项目简介:</div>
                        优化image tracking功能，提升识别、跟踪模板图像的召回率和稳定性。image tracking功能是实时识别、跟踪特定的模板图像，估计相对模板图像的pose。
                        <div class="main-work">主要工作:</div>
                        <ul>
                            <li>实现tracking模块: 通过NCC模板匹配，获得匹配点对，gpnp解算pose，联合优化重投影误差，并添加相邻帧间相对转换约束，使旋转的估计更稳定;</li>
                            <li>优化识别模块: 支持stereo识别;</li>
                        </ul>
                        <div class="achievement">项目成果:</div>
                        <ul>
                            <li>image tracking SDK文档;</li>
                            <li>CES展会: 利用image tracking功能，切换应用;</li>
                        </ul>
                    </div>
                </td>
            </tr>
        </table>

        <table class="timeline-table">
            <tr>
                <td class="period-cell">2021.7 - 2021.10</td>
                <td class="content-cell">
                    <div class="project-title">餐厅机器人面向</div>
                    <div class="company">Ninebot-Segway</div>
                    <div class="project-description">
                        <div class="main-work">项目简介:</div>
                        餐厅机器人项目，使用tag在餐厅等动态环境中为机器人提供稳定的视觉特征。
                        <div class="main-work">主要工作:</div>
                        <ul>
                            <li>将tag作为特殊的6DoF landmark添加到视觉建图框架中; tag pnp定位结果，作为fusion模块的观测;</li>
                            <li>以kalibr为code base，设计机器人顶部红外摄像头的标定SOP，交付工厂操作;</li>
                        </ul>
                    </div>
                </td>
            </tr>
        </table>

        <table class="timeline-table">
            <tr>
                <td class="period-cell">2020.5 - 2020.7</td>
                <td class="content-cell">
                    <div class="project-title">融合odom优化VIO位姿估计</div>
                    <div class="company">Ninebot-Segway</div>
                    <div class="project-description">
                        <div class="main-work">项目简介:</div>
                        针对机器人治光输运动，做匀加速运动或匀速/无旋转运动等运动特性，优化vins-mono前端;
                        <div class="main-work">主要工作:</div>
                        <ul>
                            <li>实现对IMU加速度计的bias初始化;</li>
                            <li>初始化阶段用odom位姿替代sfm过程，优化时加入odom观测，提供尺度约束和平面约束，避免匀加速运动时，尺度不可观，以及无旋转时，pitch和roll不可观的问题;</li>
                        </ul>
                    </div>
                </td>
            </tr>
        </table>
    </div>

    <div class="section">
        <div class="section-title">竞赛经历</div>
        <table class="timeline-table">
            <tr>
                <td class="period-cell">Lifelong SLAM Challenge (IROS2019 workshop)</td>
                <td class="content-cell">
                    <div class="company">竞赛名次: 第三名</div>
                </td>
            </tr>
        </table>
    </div>

    <div class="section">
        <div class="section-title">专业技能</div>
        <ul class="skills-list">
            <li>熟悉C++、python、Linux操作环境;</li>
            <li>熟悉Colmap、OpenMVG等SfM框架;</li>
            <li>熟悉vins-mono、maplab等VSLAM框架;</li>
            <li>熟悉多视图几何、非线性优化等;</li>
        </ul>
    </div>
</body>
</html>
    """
    return html_content

def generate_pdf_from_html():
    """从HTML生成PDF"""
    output_path = "/mnt/data/宋姝-SLAM-简历-WeasyPrint版.pdf"
    
    # 创建HTML内容
    html_content = create_resume_html()
    
    # 生成PDF
    try:
        # 从HTML字符串创建PDF
        weasyprint.HTML(string=html_content).write_pdf(output_path)
        print(f"简历已生成: {output_path}")
        return output_path
    except Exception as e:
        print(f"生成PDF时出错: {e}")
        return None

if __name__ == "__main__":
    generate_pdf_from_html()
