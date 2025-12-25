import matplotlib.pyplot as plt
import numpy as np

def plot_label_on_bar(bars,  ax):
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2, height + 0.2,  # +1 moves text slightly above bar
            f'{height}', ha='center', va='bottom', fontsize=20
        )


number_of_included_papers = 78

# Example data
schema_labels = [ 'Without Schema','Schema-based']
sizes = [58,  20]
colors = ['#ff9999', '#66b3ff']
#explode = (0, 0) # no explode

x = np.linspace(0, len(schema_labels) - 1, len(schema_labels))  # default spacing = 1.0 between bars
x = x * 0.35  # 🔹 reduce spacing (0.7 < 1.0 → bars closer)

# Create pie chart
schema_based_pie_fig, schema_based_ax1 = plt.subplots(figsize=(21, 15))
bars = schema_based_ax1.bar(
    x,
    sizes,
    color=colors,
    #startangle=140,
    edgecolor='black',
    width=0.3,
    #textprops={'fontsize': 20, 'family': 'DejaVu Sans', 'color': 'black'}
)
#schema_based_ax1.title.set_text("Schema Usage in Literature")

plot_label_on_bar(bars, schema_based_ax1)

schema_based_ax1.set_ylim(0, 65)
schema_based_ax1.tick_params(axis='y', labelsize=20)

schema_based_ax1.legend(bars, schema_labels, fontsize= 25, loc = "upper right")

schema_based_ax1.set_xticks(x)
schema_based_ax1.set_xticklabels([])

schema_based_pie_fig.savefig("Images/schema_based_characterization.png", dpi=100)

schema_based_pie_fig.show()
input("Press Enter to close...") 

application_labels = ["Link Prediction", "Question Answering", "Recommendation Systems", "Model proposal", "Robotics", "SPARQL Query Optimization", "Fact Explanation", "Other"]
sizes = [45, 12, 6, 3, 2, 2, 1, 7]
colors = ['#ff9999', '#66b3ff', '#99ff99', "#a45d15", '#c2c2f0', '#c4e17f', '#ffb3e6', "#921333"]
#explode = (0, 0, 0, 0, 0.05, 0)  # slightly explode the 'Fact Explanation' slice

x = np.linspace(0, len(application_labels) - 1, len(application_labels))  # default spacing = 1.0 between bars
x = x * 0.35  # 🔹 reduce spacing (0.7 < 1.0 → bars closer)

# Create pie chart
application_pie_fig, application_ax1 = plt.subplots(figsize=(21, 15))
bars = application_ax1.bar(
    x,
    sizes, 
    #labels=application_labels, 
    color=colors, 
    edgecolor='black',
    width=0.3,
    #explode=explode,
    #autopct='%1.1f%%',  # show percentages
    #shadow=False, 
    #startangle=140,
    #pctdistance=0.55,    # push % labels outward
    #labeldistance=1.05,
    #textprops={'fontsize': 20, 'family': 'DejaVu Sans', 'color': 'black'}
)

plot_label_on_bar(bars, application_ax1)

application_ax1.set_ylim(0, 65)
application_ax1.tick_params(axis='y', labelsize=20)

#application_ax1.title.set_text("Applications in Literature")
application_ax1.legend(bars, application_labels, fontsize= 25, loc = "upper right")

application_ax1.set_xticks(x)
application_ax1.set_xticklabels([])

application_pie_fig.show()
input("Press Enter to close...")

application_pie_fig.savefig("Images/application_characterization.png", dpi=100)

representation_labels = ["N-ary", "Hyper-Relational", "Compact Relations", "Metagraph", "Graph Conversion", "Other"]
sizes = [39, 23, 9, 2, 1, 4]
colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0', "#921333"]
#explode = (0, 0, 0, 0)  # no explode

x = np.linspace(0, len(representation_labels) - 1, len(representation_labels))  # default spacing = 1.0 between bars
x = x * 0.35  # 🔹 reduce spacing (0.7 < 1.0 → bars closer)

# Create pie chart
representation_pie_fig, representation_ax1 = plt.subplots(figsize=(21, 15))
bars = representation_ax1.bar(
    x,
    sizes,
    color=colors,
    edgecolor='black',
    width=0.3,
    #shadow=False, 
    #startangle=140,
    #textprops={'fontsize': 20, 'family': 'DejaVu Sans', 'color': 'black'}
)
#representation_ax1.title.set_text("Representation Types in Literature")

plot_label_on_bar(bars, representation_ax1)

representation_ax1.set_ylim(0, 65)

representation_ax1.tick_params(axis='y', labelsize=20)

representation_ax1.set_xticks(x)
representation_ax1.set_xticklabels([])

representation_ax1.legend(bars, representation_labels, fontsize= 25, loc = "upper right")

representation_pie_fig.savefig("Images/representation_characterization.png", dpi=100)

representation_pie_fig.show()
input("Press Enter to close...")
