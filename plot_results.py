"""Generate high-quality benchmark figures and architecture illustrations."""
import json
import os
os.environ["MPLCONFIGDIR"] = "/tmp/mpl"
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

STYLE = {
    'bg_fig': '#0F172A',       # Slate 900
    'bg_ax': '#1E293B',        # Slate 800
    'text_light': '#F8FAFC',   # Slate 50
    'text_muted': '#94A3B8',   # Slate 400
    'grid': '#334155',         # Slate 700
    'colors': {
        'gmlp': '#F59E0B',         # Amber
        'attention': '#38BDF8',    # Sky blue
        'amlp': '#A855F7',         # Purple
        'no_mixing': '#64748B',    # Slate
        'random_init': '#EC4899',  # Pink
    }
}

MODEL_NAMES = {
    'gmlp': 'gMLP (Static SGU)',
    'attention': 'Transformer Attn',
    'amlp': 'aMLP (gMLP + Attn)',
    'no_mixing': 'No Mixing (Ablation)',
    'random_init': 'gMLP (Random Init)'
}


def draw_architecture(output_path: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8.5), gridspec_kw={'width_ratios': [1.2, 1]})
    
    fig.patch.set_facecolor(STYLE['bg_fig'])
    ax1.set_facecolor(STYLE['bg_fig'])
    ax2.set_facecolor(STYLE['bg_fig'])
    
    # ---- Panel 1: gMLP Block & SGU Flow ----
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 14)
    ax1.axis('off')
    
    ax1.text(5, 13.5, "gMLP Block & Spatial Gating Unit (SGU)", color=STYLE['text_light'], 
             fontsize=15, fontweight='bold', ha='center')
    
    def box(x, y, w, h, text, subtext="", bg="#1E293B", border="#38BDF8", text_color="#F8FAFC"):
        rect = patches.FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle="round,pad=0.2",
                                      facecolor=bg, edgecolor=border, linewidth=1.8, zorder=2)
        ax1.add_patch(rect)
        if subtext:
            ax1.text(x, y+0.18, text, color=text_color, fontsize=10, fontweight='bold', ha='center', va='center', zorder=3)
            ax1.text(x, y-0.22, subtext, color=STYLE['text_muted'], fontsize=8.5, ha='center', va='center', zorder=3)
        else:
            ax1.text(x, y, text, color=text_color, fontsize=10, fontweight='bold', ha='center', va='center', zorder=3)

    def arrow(x1, y1, x2, y2, color="#38BDF8", style="->", rad=0.0):
        con = patches.ConnectionStyle("arc3,rad=" + str(rad)) if rad != 0 else "arc3"
        ax1.annotate("", xy=(x2, y2), xytext=(x1, y1),
                     arrowprops=dict(arrowstyle=style, color=color, lw=1.8, connectionstyle=con),
                     zorder=1)

    # Input
    box(5, 12.5, 4.0, 0.7, "Input Tensor X", "[Batch, N_tokens, d_model]", bg="#1E293B", border="#64748B")
    
    # LayerNorm
    arrow(5, 12.15, 5, 11.45)
    box(5, 11.1, 3.2, 0.6, "Layer Normalization", "Normalize over d_model", bg="#334155", border="#38BDF8")
    
    # Channel Expansion
    arrow(5, 10.8, 5, 10.1)
    box(5, 9.75, 4.4, 0.65, "Channel Projection (Linear + GeLU)", "d_model -> d_ffn (Expansion factor = 2)", bg="#1E293B", border="#818CF8")
    
    # SGU Container box
    sgu_box = patches.FancyBboxPatch((0.8, 3.5), 8.4, 5.4, boxstyle="round,pad=0.25",
                                     facecolor="#1E293B", edgecolor="#F59E0B", linewidth=2.0, linestyle="--", alpha=0.9, zorder=1)
    ax1.add_patch(sgu_box)
    ax1.text(1.2, 8.6, "Spatial Gating Unit (SGU)", color="#FBBF24", fontsize=11, fontweight='bold', zorder=2)
    
    # Split
    arrow(5, 9.42, 5, 8.65)
    box(5, 8.35, 3.2, 0.55, "Split Along Channels", "Z -> U, V (each d_ffn / 2)", bg="#0F172A", border="#F59E0B")
    
    # Branch U
    arrow(3.6, 8.07, 2.5, 5.8, color="#F59E0B")
    box(2.5, 5.5, 2.2, 0.65, "Pass-Through U", "[B, N, d_ffn/2]", bg="#0F172A", border="#FBBF24")
    
    # Branch V
    arrow(6.4, 8.07, 7.5, 7.7, color="#F59E0B")
    box(7.5, 7.4, 2.5, 0.55, "LayerNorm(V)", "Norm before gate", bg="#334155", border="#38BDF8")
    
    arrow(7.5, 7.12, 7.5, 6.4)
    box(7.5, 6.0, 3.2, 0.75, "Static Spatial Mixer (W)", "W_s @ V + b_s (N x N learned matrix)", bg="#312E81", border="#A78BFA")
    
    # Tiny Attention
    box(7.5, 4.6, 3.2, 0.55, "+ Tiny Attention (aMLP)", "1-head 16-d content attention", bg="#1E1B4B", border="#C084FC", text_color="#E9D5FF")
    arrow(7.5, 5.62, 7.5, 4.88, color="#C084FC")
    
    # Element-wise gate
    arrow(2.5, 5.17, 4.5, 4.0, color="#FBBF24")
    arrow(7.5, 4.32, 5.5, 4.0, color="#C084FC")
    box(5, 3.9, 2.4, 0.6, "Gate: U ⊙ V_gate", "Elementwise multiply", bg="#B45309", border="#FDE68A")
    
    # Channel Contraction
    arrow(5, 3.5, 5, 2.7)
    box(5, 2.35, 4.2, 0.65, "Output Linear Projection", "d_ffn / 2 -> d_model", bg="#1E293B", border="#818CF8")
    
    # Residual path
    ax1.plot([2.0, 0.5, 0.5, 2.8], [12.5, 12.5, 1.2, 1.2], color="#38BDF8", lw=1.8, linestyle=":", zorder=1)
    arrow(2.8, 1.2, 4.5, 1.2, color="#38BDF8")
    arrow(5, 2.02, 5, 1.45)
    
    # Add + Norm/Out
    box(5, 1.2, 1.4, 0.5, "Add (+)", "Residual", bg="#0F172A", border="#38BDF8")
    arrow(5, 0.95, 5, 0.5)
    box(5, 0.35, 3.8, 0.55, "Output Tensor", "[Batch, N_tokens, d_model]", bg="#1E293B", border="#64748B")

    # ---- Panel 2: Static Mixing vs Dynamic Attention ----
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 14)
    ax2.axis('off')
    
    ax2.text(5, 13.5, "Static Mixing vs Dynamic Attention", color=STYLE['text_light'], 
             fontsize=15, fontweight='bold', ha='center')
    
    # Top card: Static Spatial Matrix (gMLP)
    card1 = patches.FancyBboxPatch((0.5, 7.2), 9.0, 5.7, boxstyle="round,pad=0.25",
                                   facecolor="#1E293B", edgecolor="#A78BFA", linewidth=2.0)
    ax2.add_patch(card1)
    ax2.text(5, 12.4, "gMLP: Static Token Routing (W_spatial)", color="#C4B5FD", 
             fontsize=12, fontweight='bold', ha='center')
    ax2.text(5, 11.75, "Learned N x N weights, FROZEN across all inputs", color=STYLE['text_muted'], 
             fontsize=9.5, ha='center', style='italic')
    
    for i in range(4):
        for j in range(4):
            val_color = "#4C1D95" if (i==0 and j==2) else "#334155"
            border_c = "#A78BFA" if (i==0 and j==2) else "#475569"
            p = patches.Rectangle((1.5 + j*0.65, 8.8 + (3-i)*0.65), 0.55, 0.55, 
                                  facecolor=val_color, edgecolor=border_c, lw=1.2)
            ax2.add_patch(p)
    ax2.text(4.5, 10.4, "Weight matrix W_s:", color="#E2E8F0", fontsize=10, fontweight='bold')
    ax2.text(4.5, 9.8, "• Fixed connection per position pair (i, j)", color="#CBD5E1", fontsize=9)
    ax2.text(4.5, 9.2, "• Solves fixed-position routing perfectly (100%)", color="#4ADE80", fontsize=9, fontweight='bold')
    ax2.text(4.5, 8.6, "• Fails when key token moves OOD (53.2% acc)", color="#F87171", fontsize=9, fontweight='bold')
    ax2.text(5, 7.6, "Insight: gMLP memorizes spatial coordinates, not token content!", 
             color="#FDE047", fontsize=9.2, ha='center', fontweight='bold')

    # Bottom card: Content Attention (Transformer / aMLP)
    card2 = patches.FancyBboxPatch((0.5, 0.8), 9.0, 5.7, boxstyle="round,pad=0.25",
                                   facecolor="#1E293B", edgecolor="#38BDF8", linewidth=2.0)
    ax2.add_patch(card2)
    ax2.text(5, 6.0, "Attention / aMLP: Dynamic Content Routing", color="#7DD3FC", 
             fontsize=12, fontweight='bold', ha='center')
    ax2.text(5, 5.35, "Softmax(Q K^T / √d): Adaptive routing per instance", color=STYLE['text_muted'], 
             fontsize=9.5, ha='center', style='italic')
             
    for i in range(4):
        for j in range(4):
            val_color = "#0369A1" if (i==0 and j==3) else "#334155"
            border_c = "#38BDF8" if (i==0 and j==3) else "#475569"
            p = patches.Rectangle((1.5 + j*0.65, 2.4 + (3-i)*0.65), 0.55, 0.55, 
                                  facecolor=val_color, edgecolor=border_c, lw=1.2)
            ax2.add_patch(p)
    ax2.text(4.5, 4.0, "Dynamic Matrix A(X):", color="#E2E8F0", fontsize=10, fontweight='bold')
    ax2.text(4.5, 3.4, "• Query & Key matching: Content-dependent", color="#CBD5E1", fontsize=9)
    ax2.text(4.5, 2.8, "• Solves fixed & content-based routing (100%)", color="#4ADE80", fontsize=9, fontweight='bold')
    ax2.text(4.5, 2.2, "• Generalizes perfectly to unseen positions (100%)", color="#4ADE80", fontsize=9, fontweight='bold')
    ax2.text(5, 1.2, "Insight: Tiny attention (aMLP) restores positional invariance!", 
             color="#67E8F9", fontsize=9.2, ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"Generated {output_path}")


def draw_benchmark_summary(summaries, output_path: Path):
    tasks = [
        ('fixed', 'Task 1: Fixed Position\n(Target always at pos 3)'),
        ('content', 'Task 2: Content (In-Dist)\n(Target marked, all pos seen)'),
        ('content_ood', 'Task 3: Content (Out-of-Dist)\n(Target moves to unseen pos 7-11)')
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharey=True)
    fig.patch.set_facecolor(STYLE['bg_fig'])
    
    for ax, (task_key, task_title) in zip(axes, tasks):
        ax.set_facecolor(STYLE['bg_ax'])
        rows = [s for s in summaries if s['task'] == task_key]
        
        # Sort or preserve order
        names = [MODEL_NAMES.get(x['model'], x['model']) for x in rows]
        values = [x['accuracy_mean'] * 100 for x in rows]
        errors = [x['accuracy_std'] * 100 for x in rows]
        colors = [STYLE['colors'].get(x['model'], '#CBD5E1') for x in rows]
        
        bars = ax.bar(names, values, yerr=errors, capsize=4, color=colors, width=0.55, edgecolor='#FFFFFF', linewidth=0.5, alpha=0.92)
        
        # Add values on top of bars
        for bar, val in zip(bars, values):
            y_pos = min(val + 3.0, 103)
            ax.text(bar.get_x() + bar.get_width() / 2, y_pos, f"{val:.1f}%", 
                    ha='center', va='bottom', color=STYLE['text_light'], fontsize=9.5, fontweight='bold')
            
        ax.set_title(task_title, color=STYLE['text_light'], fontsize=12, fontweight='bold', pad=12)
        ax.set_ylim(0, 115)
        ax.grid(axis='y', linestyle='--', alpha=0.3, color=STYLE['grid'])
        ax.tick_params(colors=STYLE['text_muted'], which='both')
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=25, ha='right', fontsize=9, color=STYLE['text_light'])
        
        # 50% baseline line
        ax.axhline(50, color='#EF4444', linestyle=':', linewidth=1.2, alpha=0.7)
        if task_key == 'content_ood':
            ax.text(2.5, 52, "Random chance (50%)", color='#FCA5A5', fontsize=8, style='italic')

    axes[0].set_ylabel('Held-out Accuracy (%)', color=STYLE['text_light'], fontsize=11, fontweight='bold')
    
    # Add subtitle banner
    fig.suptitle("gMLP vs Transformer Baselines on Synthetic Token-Retrieval Tasks", 
                 color=STYLE['text_light'], fontsize=15, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"Generated {output_path}")


def draw_individual_plots(summaries, root: Path):
    titles = {
        'fixed': 'Fixed-Position Retrieval Accuracy (Always Pos 3)',
        'content': 'Content-Based Retrieval Accuracy (In-Distribution)',
        'content_ood': 'Content-Based Retrieval Accuracy (Unseen Pos 7-11)'
    }
    
    for task in ('fixed', 'content', 'content_ood'):
        rows = [s for s in summaries if s['task'] == task]
        names = [MODEL_NAMES.get(x['model'], x['model']) for x in rows]
        values = [x['accuracy_mean'] * 100 for x in rows]
        errors = [x['accuracy_std'] * 100 for x in rows]
        colors = [STYLE['colors'].get(x['model'], '#CBD5E1') for x in rows]
        
        fig, ax = plt.subplots(figsize=(8, 4.8))
        fig.patch.set_facecolor(STYLE['bg_fig'])
        ax.set_facecolor(STYLE['bg_ax'])
        
        bars = ax.bar(names, values, yerr=errors, capsize=4, color=colors, width=0.5, edgecolor='#FFFFFF', linewidth=0.5, alpha=0.92)
        
        for bar, val in zip(bars, values):
            y_pos = min(val + 3.0, 103)
            ax.text(bar.get_x() + bar.get_width() / 2, y_pos, f"{val:.1f}%", 
                    ha='center', va='bottom', color=STYLE['text_light'], fontsize=10, fontweight='bold')
            
        ax.axhline(50, color='#EF4444', linestyle=':', linewidth=1.2, alpha=0.7)
        ax.text(len(names)-1, 52, "Random chance (50%)", color='#FCA5A5', fontsize=8.5, style='italic', ha='right')
        
        ax.set_ylim(0, 115)
        ax.set_ylabel('Held-out Accuracy (%)', color=STYLE['text_light'], fontsize=11, fontweight='bold')
        ax.set_title(titles[task], color=STYLE['text_light'], fontsize=13, fontweight='bold', pad=12)
        ax.grid(axis='y', linestyle='--', alpha=0.3, color=STYLE['grid'])
        ax.tick_params(colors=STYLE['text_muted'], which='both')
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=15, ha='right', fontsize=9.5, color=STYLE['text_light'])
        
        plt.tight_layout()
        out_path = root / f"figures/{task}_accuracy.png"
        plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close()
        print(f"Generated {out_path}")


def main():
    root = Path(__file__).resolve().parent
    data = json.loads((root / 'results/benchmark.json').read_text())
    summaries = data['summary']
    figures_dir = root / 'figures'
    figures_dir.mkdir(exist_ok=True)
    
    # 1. Architecture diagram
    draw_architecture(figures_dir / "architecture_diagram.png")
    
    # 2. Combined summary figure
    draw_benchmark_summary(summaries, figures_dir / "benchmark_summary.png")
    
    # 3. Individual updated task plots
    draw_individual_plots(summaries, root)


if __name__ == '__main__':
    main()
