import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.path import Path

plt.rcParams.update({"font.family": "DejaVu Sans", "figure.dpi": 300})

fig, ax = plt.subplots(figsize=(9.5, 6.2))
ax.set_xlim(0, 10)
ax.set_ylim(0, 6.4)
ax.axis("off")

def box(x, y, w, h, text, fc, ec="#1b4965", fs=10.2, tc="white", weight="bold"):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                        linewidth=1.4, edgecolor=ec, facecolor=fc)
    ax.add_patch(b)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs,
             color=tc, weight=weight, linespacing=1.35)
    return (x, y, w, h)

def arrow(p1, p2, color="#1b4965", style="-", lw=2.2, curve=0.0):
    a = FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=18, linewidth=lw,
                         color=color, linestyle=style,
                         connectionstyle=f"arc3,rad={curve}")
    ax.add_patch(a)

# Main constructs
b_tour = box(0.3, 4.3, 2.6, 1.5, "Tourism Development &\nTourist Behavior\n(arrivals, receipts)", "#1b4965")
b_env  = box(3.75, 4.3, 2.6, 1.5, "Environmental\nDegradation\n(CO2, PM2.5)", "#457b9d")
b_health = box(7.2, 4.3, 2.5, 1.5, "Public Health &\nWellbeing\n(life expectancy)", "#2a6f97")

# Digital transformation / emerging tech (moderator)
b_tech = box(2.9, 1.1, 4.4, 1.4, "Digital Transformation & Emerging Technologies\n(broadband/AI infrastructure, smart systems, GenAI)",
             "#c9184a", fs=9.6)

# Controls box
b_ctrl = box(0.3, 0.15, 9.4, 0.7, "Controls: GDP per capita | Urbanization | Renewable energy adoption | Population",
             "#e9ecef", ec="#adb5bd", tc="#212529", fs=9, weight="normal")

# H1: Tourism -> Environment
arrow((2.9, 5.05), (3.75, 5.05))
ax.text(3.32, 5.25, "H1", fontsize=10, weight="bold", color="#1b4965", ha="center")

# H2: Environment -> Health
arrow((6.35, 5.05), (7.2, 5.05))
ax.text(6.78, 5.25, "H2", fontsize=10, weight="bold", color="#1b4965", ha="center")

# H3: Tech moderates H1 (dashed, pointing up into the H1 arrow midpoint)
arrow((4.2, 2.5), (3.32, 4.28), color="#c9184a", style="--", lw=1.8, curve=0.15)
ax.text(3.55, 3.15, "H3", fontsize=10, weight="bold", color="#c9184a", ha="center", rotation=0)

# H4: Tech moderates H2
arrow((5.9, 2.5), (6.78, 4.28), color="#c9184a", style="--", lw=1.8, curve=-0.15)
ax.text(6.55, 3.15, "H4", fontsize=10, weight="bold", color="#c9184a", ha="center")

# Controls feeding into all
arrow((2.4, 0.85), (2.4, 4.28), color="#adb5bd", lw=1.3, style=":")
arrow((5.0, 0.85), (5.0, 4.28), color="#adb5bd", lw=1.3, style=":")
arrow((7.9, 0.85), (7.9, 4.28), color="#adb5bd", lw=1.3, style=":")

ax.set_title("Figure 1. Conceptual Framework: The Dynamic Tourism-Environment-Health-Technology (TEHT) Nexus",
             fontsize=11.5, weight="bold", pad=14)

fig.tight_layout()
fig.savefig("/home/claude/paper/figures/fig1_conceptual_framework.png", bbox_inches="tight", dpi=300)
print("Figure 1 saved.")
