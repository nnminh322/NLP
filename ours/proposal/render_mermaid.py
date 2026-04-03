"""Extract mermaid blocks from beauty_architecture.md and render to PNG."""
import re
import subprocess
import os
from pathlib import Path

ARCH_MD = Path(__file__).parent.parent / "source" / "beauty_architecture.md"
OUT_DIR = Path(__file__).parent / "architecture"
OUT_DIR.mkdir(exist_ok=True)

content = ARCH_MD.read_text(encoding="utf-8")

# Extract all mermaid code blocks
pattern = re.compile(r'## (Figure \d+\w?) — (.+?)\n.*?```mermaid\n(.*?)```', re.DOTALL)
matches = pattern.findall(content)

filenames = {
    "Figure 1": "fig1_framework",
    "Figure 2": "fig2_kg_construction",
    "Figure 3": "fig3_gat_encoder",
    "Figure 3b": "fig3b_edge_attention",
    "Figure 4": "fig4_joint_scorer",
    "Figure 5": "fig5_chap",
    "Figure 6": "fig6_cacl_loss",
    "Figure 7": "fig7_curriculum",
}

for fig_id, title, mermaid_code in matches:
    fig_key = fig_id.strip()
    fname = filenames.get(fig_key, fig_key.lower().replace(" ", "_"))
    mmd_path = OUT_DIR / f"{fname}.mmd"
    png_path = OUT_DIR / f"{fname}.png"
    
    mmd_path.write_text(mermaid_code.strip(), encoding="utf-8")
    print(f"Rendering {fig_key}: {title} -> {png_path.name}")
    
    result = subprocess.run(
        ["mmdc", "-i", str(mmd_path), "-o", str(png_path), 
         "-w", "1600", "-b", "white", "-s", "2"],
        capture_output=True, text=True, timeout=60
    )
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr[:200]}")
    else:
        print(f"  OK: {png_path.name}")
    
    # Clean up mmd file
    mmd_path.unlink()

print("\nDone. Files in", OUT_DIR)
for f in sorted(OUT_DIR.glob("*.png")):
    print(f"  {f.name} ({f.stat().st_size // 1024} KB)")
