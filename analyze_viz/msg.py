import streamlit as st

st.set_page_config(page_title="群消息模板生成器", layout="centered")

st.title("📣 群消息模板生成器")

msg_type = st.selectbox("请选择消息类型", [
    "代码变更通知",
    "功能/测试完成",
    "异常/Bug 提醒",
    "数据处理通报",
    "工具脚本说明"
])

output = ""

if msg_type == "代码变更通知":
    project = st.text_input("项目名称")
    change = st.text_area("变更内容")
    notes = st.text_area("注意事项（每行一项）")
    link = st.text_input("PR 链接（可选）")
    who = st.text_input("需确认人 @xxx")

    if st.button("生成消息"):
        output = f"""
📌【代码库更新通知】
项目：{project}
变更内容：已完成 {change}
注意事项：
"""
        for note in notes.split("\n"):
            output += f"- {note}\n"
        if link:
            output += f"📎 PR链接：{link}\n"
        output += f"\n👉 请 {who} 确认是否有依赖影响，今天下班前 @我确认一下"

elif msg_type == "功能/测试完成":
    name = st.text_input("功能/测试名称")
    env = st.text_input("测试环境（如 staging/dev）")
    result = st.text_area("测试/实现结果（每行一项）")
    advice = st.text_area("建议（可选）")
    link = st.text_input("报告链接（可选）")

    if st.button("生成消息"):
        output = f"""
✅【功能/测试进展通报】
已完成：{name}
环境：{env}
输出结论：
"""
        for line in result.split("\n"):
            output += f"- {line}\n"
        if advice:
            output += f"👉 建议：{advice}\n"
        if link:
            output += f"📎 报告链接：{link}"

elif msg_type == "异常/Bug 提醒":
    problem = st.text_input("问题描述")
    scope = st.text_input("影响范围")
    finding = st.text_area("排查结论（每行一项）")
    solution = st.text_area("临时方案/下一步（可选）")
    link = st.text_input("日志/截图链接（可选）")
    who = st.text_input("相关人 @xxx（可选）")

    if st.button("生成消息"):
        output = f"""
🚨【异常/问题提醒】
问题描述：{problem}
影响范围：{scope}
初步排查结论：
"""
        for line in finding.split("\n"):
            output += f"- {line}\n"
        if solution:
            output += f"👉 当前处理建议：{solution}\n"
        if link:
            output += f"📎 相关链接：{link}\n"
        if who:
            output += f"👉 已通知：{who}"

elif msg_type == "数据处理通报":
    data = st.text_input("数据范围")
    method = st.text_area("处理方式")
    result = st.text_area("分析结果（每行一项）")
    nextplan = st.text_area("下一步计划")
    link = st.text_input("数据或图表链接（可选）")

    if st.button("生成消息"):
        output = f"""
📊【数据处理通报】
已处理数据：{data}
处理方式：{method}
输出结论：
"""
        for line in result.split("\n"):
            output += f"- {line}\n"
        output += f"下一步计划：{nextplan}\n"
        if link:
            output += f"📎 链接：{link}"

elif msg_type == "工具脚本说明":
    name = st.text_input("工具名称")
    desc = st.text_area("功能简述")
    cmd = st.text_area("使用示例命令")
    notes = st.text_area("注意事项（每行一项）")
    link = st.text_input("脚本路径或文档链接")

    if st.button("生成消息"):
        output = f"""
🛠️【工具脚本发布】
名称：{name}
功能简述：{desc}
使用方式：
```bash
{cmd}
```
注意事项：
"""
        for line in notes.split("\n"):
            output += f"- {line}\n"
        output += f"📎 链接：{link}"

if output:
    st.markdown("---")
    st.subheader("📤 生成的群消息内容")
    st.code(output, language="markdown")
