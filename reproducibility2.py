from RecSysFramework.Utils import menu
from RecSysFramework.Experiments.difference_of_recs_among_same_models import difference_of_recs_among_same_models
from RecSysFramework.Experiments.generate_performance_table import generate_performance_table
from RecSysFramework.Experiments.recommended_items_popularity_analysis import recommended_items_popularity_analysis
from RecSysFramework.Experiments.statistically_significance import statistically_significance
from RecSysFramework.Experiments.plot_stability_short_head import plot_stability_short_head
from RecSysFramework.Experiments.convergence import plot

if __name__ == "__main__":
    """This file is aimed at reproducing the results of the experiments. 
       It requires to have the best models saved in BestModels 
       (see the other reproducibility script)
    """

    # Run the stability tests.
    while True:
        choice = menu.yesno_choice("Do you want to run a stability test?")
        if choice == 'y':
            difference_of_recs_among_same_models()
        else:
            break

    # Compute metrics on test set
    while True:
        choice = menu.yesno_choice("Do you want to compute metrics?")
        if choice == 'y':
            generate_performance_table()
        else:
            break

    # Statistically significance
    while True:
        choice = menu.yesno_choice(
            "Do you want carry out significance test for the metrics MF against NNMF?")
        if choice == 'y':
            statistically_significance()
        else:
            break

    # Recommended items popularity analysis
    while True:
        choice = menu.yesno_choice(
            "Do you want to compute statistics the popularity of items recommended?")
        if choice == 'y':
            recommended_items_popularity_analysis()
        else:
            break

    # Stability of representations varying item popularity bin
    while True:
        choice = menu.yesno_choice(
            "Do you want to plot the stability of representations varying item popularity bin?")
        if choice == 'y':
            plot_stability_short_head()
        else:
            break

    # Plot convergence
    while True:
        choice = menu.yesno_choice(
            "Do you want to plot algorithms loss and MAP varying epochs?")
        if choice == 'y':
            plot()
        else:
            break
