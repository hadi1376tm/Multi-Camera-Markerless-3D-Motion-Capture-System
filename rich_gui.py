from rich.console import Console
from rich.layout import Layout

console = Console()
layout = Layout()

layout.split(
    Layout(name="header", size=3),
    Layout(ratio=1, name="main"),
    Layout(size=10, name="footer"),
)
layout["main"].split(
    Layout(name="side"),
    Layout(name="body", ratio=2),
    direction="horizontal"
)
layout["side"].split(Layout(), Layout())

console.print(layout)