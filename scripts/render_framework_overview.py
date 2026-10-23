#!/usr/bin/env python3
"""Render the manuscript-aligned framework as portable vector and raster assets."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, PathPatch
from matplotlib.path import Path as PlotPath

BLUE, GREEN, ORANGE, PURPLE, INK = '#315f8b', '#397849', '#a45b2a', '#7c5198', '#252525'

def panel(ax, x, y, w, h, color, fill='white', radius=.10, linewidth=1.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0.02,rounding_size={radius}',
                              facecolor=fill, edgecolor=color, linewidth=linewidth))

def text(ax, x, y, value, size=12, color=INK, weight='normal', ha='center', va='center'):
    return ax.text(x, y, value, fontsize=size, color=color, fontweight=weight, ha=ha, va=va,
                   linespacing=1.35)

def arrow(ax, a, b, color=INK, dashed=False, width=1.7):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=15,
                               color=color, linewidth=width, linestyle='--' if dashed else '-'))

def render(facts, destination):
    plt.rcParams.update({'font.family': 'DejaVu Serif', 'mathtext.fontset': 'stix',
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    fig, ax = plt.subplots(figsize=(16, 8.2))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set(xlim=(0,16), ylim=(0,8.2)); ax.axis('off')
    panel(ax, .12, 4.43, 15.76, 3.65, '#a3a3a3', '#f5f5f5', .17)
    text(ax, .35, 7.78, 'AUGMENTED GAIA: DEPENDENCY-AWARE REFERENCE CONSTRUCTION', 16,
         weight='bold', ha='left')

    # Native task and its group-specific tool environment.
    panel(ax, .32, 5.42, 2.55, 1.94, '#8a8a8a')
    text(ax, 1.59, 7.12, 'Native GAIA task', 14, weight='bold')
    text(ax, 1.59, 6.63, 'Text · Document\nVision · Audio', 12)
    for i, label in enumerate(['0','1','2','n']):
        x = .64 + i*.62
        ax.add_patch(Circle((x,6.03), .16, facecolor='white', edgecolor='#555', linewidth=1.5))
        text(ax,x,6.03,label,11)
        if i<3: arrow(ax,(x+.16,6.03),(x+.46,6.03), '#555', width=1.1)
    text(ax,1.59,5.65,f"{facts['tasks']} tasks · native chains",11)
    panel(ax,.32,4.91,2.55,.35,GREEN,'#eaf3e6',.05)
    text(ax,1.59,5.085,f"Tool layer: {facts['gold_tool_slots']} gold slots / {facts['tool_annotated_tasks']} tasks",9.5,GREEN)
    text(ax,1.59,4.70,f"{facts['open_weight_catalog_actions']} actions (open) · {facts['closed_model_catalog_actions']} (closed)",9.5,GREEN)

    panel(ax,3.25,5.42,2.45,1.94,PURPLE,'#eee4f2')
    text(ax,4.475,6.96,'AD/SD annotation',14,PURPLE,'bold')
    text(ax,4.475,6.37,'LLM witnesses +\ndeterministic compiler',11.5)
    ad,sd=facts['artifact_dependency_records'],facts['state_dependency_records']
    text(ax,4.475,5.76,f'{ad:,} AD + {sd:,} SD\n{ad+sd:,} semantic records',11)
    arrow(ax,(2.91,6.35),(3.20,6.35))

    panel(ax,6.07,5.42,2.31,1.94,'#8a8a8a')
    text(ax,7.225,7.05,'Dependency DAG',13,weight='bold')
    positions=[(6.43,6.35),(7.2,6.72),(7.2,5.99),(7.98,6.35)]
    for a,b in [(0,1),(0,2),(1,3),(2,3)]:
        p,q=positions[a],positions[b]
        ax.add_patch(FancyArrowPatch(p,q,shrinkA=13,shrinkB=13,arrowstyle='-|>',
                                   mutation_scale=11,linewidth=1.6,color='#218d86'))
    for i,p in enumerate(positions):
        ax.add_patch(Circle(p,.16,facecolor='white',edgecolor='#555',linewidth=1.4))
        text(ax,*p,str(i),11)
    text(ax,7.225,5.61,f"{facts['native_nodes']:,} nodes\n{facts['reduced_precedence_edges']:,} reduced edges",9.5)
    arrow(ax,(5.74,6.35),(6.02,6.35))

    panel(ax,8.75,5.42,2.64,1.94,'#8a8a8a')
    text(ax,10.07,7.04,'Recorded orderings',13,weight='bold')
    text(ax,10.07,6.49,'[0, 1, 2, 3]   Native\n[0, 2, 1, 3]   candidate\n…',11.5)
    text(ax,10.07,5.78,f"{facts['non_native_candidates']:,} non-native candidates\n{facts['tasks_with_candidates']} tasks with alternatives",10.5)
    arrow(ax,(8.42,6.35),(8.70,6.35))

    panel(ax,11.79,5.42,3.74,1.94,BLUE,'#e2ebf6')
    text(ax,13.66,7.02,'Main reference set',14,BLUE,'bold')
    text(ax,13.66,6.48,f"{facts['native_references']} Native + {facts['non_native_candidates']:,} candidates\n{facts['total_references']:,} references in total",13)
    text(ax,13.66,5.81,f"Full unfiltered pool\n{facts['multi_order_rich_tasks']} multi-order-rich tasks",11,BLUE)
    arrow(ax,(11.43,6.35),(11.74,6.35),BLUE)
    text(ax,8.0,4.70,'Replay + anonymous human AD/SD review provide separate quality evidence; they do not filter the main reference set.',10.6)

    # Lower row: pre-execution planning, realized tools, group-specific response.
    panel(ax,.12,.12,15.76,3.77,'#278d88','#eff7f6',.17)
    text(ax,.35,3.60,'STAGED INFERENCE · THREE-FACET EVALUATION',15,'#226e6a','bold',ha='left')
    columns=[(2.35,4.0,BLUE,'(A) Plan structural fidelity',
              'S3 pre-execution graph vs. best reference',
              'NodeF1 · EdgeF1',r'PlanningScore = (NodeF1 + EdgeF1) / 2'),
             (6.75,4.0,GREEN,'(B) Tool invocation accuracy',
              'Realized calls vs. gold tool slots',
              'ToolF1 · ParamF1 · ParamValueF1','ToolUsageScore = mean of the three'),
             (11.15,4.35,ORANGE,'(C) Final-answer correctness',
              'Returned response vs. reference answer',
              'Norm. EM · TokenF1 · Accepted','Headline: human Accepted Answer rate')]
    for x,w,c,title,sub,metrics,formula in columns:
        panel(ax,x,1.93,w,1.29,c,{'#315f8b':'#e5edf7','#397849':'#e8f2e6'}.get(c,'#f8ede2'))
        text(ax,x+w/2,2.99,title,12,c,'bold')
        text(ax,x+w/2,2.64,sub,10.2)
        text(ax,x+w/2,2.30,metrics,11.2,weight='bold')
        text(ax,x+w/2,2.06,formula,9.3)
    # Simple vector agent icon.
    panel(ax,.52,2.17,1.12,.71,'#666','#c7c7c7',.12)
    for x in [.82,1.34]:
        ax.add_patch(Circle((x,2.61),.12,facecolor='white',edgecolor='#666',linewidth=1.2))
        ax.add_patch(Circle((x,2.61),.045,facecolor='#333'))
    ax.plot([1.08,1.08],[2.90,3.08],color='#666',linewidth=2)
    ax.add_patch(Circle((1.08,3.10),.065,facecolor='#888',edgecolor='#666'))
    text(ax,1.08,1.87,'Agent under test',11,weight='bold')
    text(ax,1.08,1.47,'Task + shared\naction catalog',10)

    blocks=[(2.35,4.0,BLUE,'Planning · S1–S3','Abstract plan → tool assessment\n→ concrete pre-execution plan'),
            (6.75,4.0,GREEN,'Execution · S4','Tool calls ↔ observations\nActual arguments and outputs retained'),
            (11.15,4.35,ORANGE,'Returned answer','Open: synthesized S5 response\nClosed: submission / final fallback')]
    for i,(x,w,c,title,body) in enumerate(blocks):
        panel(ax,x,.55,w,.97,c)
        text(ax,x+w/2,1.28,title,13,c,'bold')
        text(ax,x+w/2,.88,body,10.6)
        arrow(ax,(x+w/2,1.55),(x+w/2,1.90),c)
        if i<2:arrow(ax,(x+w+.03,1.02),(blocks[i+1][0]-.05,1.02))
    arrow(ax,(1.70,1.02),(2.30,1.02))
    text(ax,8.10,.31,'S3 is primary; S1 is auxiliary. Predictions and returned answers stay fixed when planning references change.',10.4)
    vertices=[(13.66,5.38),(13.66,4.10),(4.35,4.10),(4.35,3.25)]
    ax.add_patch(PathPatch(PlotPath(vertices,[PlotPath.MOVETO,*[PlotPath.LINETO]*3]),
                           fill=False,color=BLUE,linewidth=1.5,linestyle='--'))
    arrow(ax,(4.35,3.40),(4.35,3.25),BLUE)
    text(ax,9.05,4.15,'Max over Native and every candidate; Native wins ties',10.5,BLUE,va='bottom')
    destination.parent.mkdir(parents=True,exist_ok=True)
    for suffix in ['png','pdf','svg']:
        metadata = ({'Software':None} if suffix=='png' else
                    {'Creator':None,'Producer':None,'Author':None,'CreationDate':None,'ModDate':None} if suffix=='pdf' else
                    {'Date':None,'Creator':None})
        fig.savefig(destination.with_suffix('.'+suffix),dpi=200,facecolor='white',metadata=metadata)
    plt.close(fig)

if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--facts',type=Path,default=root/'docs/assets/framework_overview.facts.json')
    parser.add_argument('--output',type=Path,default=root/'docs/assets/framework_overview')
    args=parser.parse_args()
    render(json.loads(args.facts.read_text()),args.output)
