from fpdf import FPDF
from fpdf.enums import XPos, YPos
import os

# 注册中文字体，优先尝试系统常见中文字体（优先 OTF/TTF）
def register_chinese_font(pdf: FPDF) -> str:
    candidate_paths = [
        "/usr/share/fonts/truetype/noto/NotoSansSC-Regular.otf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
        "/usr/share/fonts/truetype/arphic/ukai.ttc",
        "/usr/share/fonts/truetype/arphic/uming.ttc",
    ]
    for font_path in candidate_paths:
        if os.path.isfile(font_path):
            pdf.add_font("CN", "", font_path)
            return "CN"
    raise RuntimeError(
        "未找到可用的中文字体，请安装 'fonts-noto-cjk' 或 'fonts-wqy-zenhei' 后重试"
    )

# 创建PDF类
class PDF(FPDF):
    def header(self):
        self.set_font("CN", "", 16)
        self.cell(0, 10, "宋姝｜SLAM算法工程师", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
        self.set_font("CN", "", 11)
        # 移除表情符号以避免字体缺字问题
        self.cell(0, 8, "手机 17801020905 ｜ 邮箱 songshu0905@gmail.com", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
        self.ln(5)

    def chapter_title(self, title):
        self.set_font("CN", "", 13)
        self.cell(0, 8, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(1)

    def chapter_body(self, body):
        self.set_font("CN", "", 11)
        self.multi_cell(0, 6, body)
        self.ln()

pdf = PDF()
font_family = register_chinese_font(pdf)
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()

# 数据
resume_sections = [
    ("石头科技｜算法工程师  2024.XX – 至今",
    """项目：室外割草机双目 vSLAM 研发与多传感器融合优化
- 算法研发与架构优化
  - 参与 vSLAM 框架重构并完成上机部署，架构由同步流程演进为异步多线程结构，运行效率提升 30%。
  - 实现 IMU + 轮速多传感器融合，显著提升定位精度与系统鲁棒性。
- 工具链与数据验证
  - 自主开发多传感器数据分析 Toolkit，支持 RTK、双目、IMU、轮速的验证、预处理与可视化分析。
  - 工具已在多轮实地测试中稳定应用，显著缩短问题排查周期。
- 标定与算法优化支持
  - 参与双目 + IMU 标定流程设计与优化，开发外参验证与误差评估工具。
  - 优化标定算法，提高标定精度与一致性。

成果亮点
- 掌握双目 vSLAM 全链路（视觉前端、非线性优化、多传感器融合、多线程架构）。
- 推动算法、工程与验证一体化，确保算法模块可落地部署。
- 在标定、跨团队调试及系统验证中提供关键技术支持。
"""),

    ("XREAL｜SLAM算法工程师  2022.11 – 2024.XX",
    """项目：Room-scale 视觉惯性建图与定位（Spatial Anchor）
- 从0到1研发跨session/跨user地图共享与定位功能，支持Linux/Android多平台API。
- 双目建图与定位：stereo matcher、stereo-IMU联合优化、GPnP双目位姿求解。
- 多种2D-3D匹配策略（active search、两阶段匹配、PQ量化），回环召回率提升XX%。
- 优化 lo-ransac + 位姿先验，提升定位精度；IMU预积分模块减少XX%优化时间。
- 多地图管理（互助定位/合并/更新）、地图质量评估、跨平台上线。
成果：集成至XREAL SDK并于CES展会展示虚拟多屏交互。
"""),

    ("Ninebot-Segway｜SLAM算法工程师  2019.XX – 2022.XX",
    """项目：City-scale 大规模室外建图
- 实现 hypermap pipeline（关键帧筛选、odom+global SfM恢复尺度、错误匹配过滤等）。
- 分块-合并建图提升大规模场景下效率与鲁棒性；融合RTK数据约束，精度从米级提升至分米级。
- 搭建benchmark评估传统与learning特征性能。
成果：ICRA论文发表，建图时间缩短至Colmap的1/5。

项目：餐厅机器人视觉定位
- 将tag作为6DoF landmark融入视觉建图框架，pnp定位结果接入融合模块。
- 设计红外摄像头标定SOP并交付工厂操作。

项目：融合odom优化VIO位姿估计
- 实现IMU加速度计bias初始化；引入odom观测约束解决不可观问题。
"""),

    ("竞赛与荣誉",
    """- IROS Lifelong SLAM Challenge 第三名"""),

    ("技能",
    """- 语言：C++（精通）、Python（熟练）
- 算法框架：vins-mono, maplab, Colmap, OpenMVG
- 技术领域：多视图几何、非线性优化、双目/IMU融合、回环检测、地图管理
- 工具链：Linux, Git, CMake, OpenCV, Eigen""")
]

# 添加章节
for title, body in resume_sections:
    pdf.chapter_title(title)
    pdf.chapter_body(body)

# 输出PDF
output_path = "/mnt/data/宋姝-SLAM-简历-优化版.pdf"
pdf.output(output_path)
output_path

