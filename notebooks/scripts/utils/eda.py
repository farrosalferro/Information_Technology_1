import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import plotly.express as px
import matplotlib.colors as mcolors
from matplotlib.cm import get_cmap

def display_dataset_samples(df, train_dir, dataset_name, n_scenes_to_show=2, n_samples_per_scene=3):
    dataset_labels = df[df['dataset'] == dataset_name]
    scenes = dataset_labels['scene'].unique()
    print(f"Scenes in {dataset_name}: {scenes}")

    fig = plt.figure(figsize=(15, 5 * n_scenes_to_show), facecolor='#f0f0f5')
    
    # Generate vibrant colors for scene boxes
    scene_colors = plt.cm.tab10(np.linspace(0, 1, len(scenes)))
    
    plot_idx = 1

    # Show samples from regular scenes
    regular_scenes = [s for s in scenes if s != 'outliers'][:n_scenes_to_show]
    for i, scene in enumerate(regular_scenes):
        scene_images = dataset_labels[dataset_labels['scene'] == scene]['image'].sample(min(n_samples_per_scene, len(dataset_labels[dataset_labels['scene'] == scene]))).tolist()
        for img_name in scene_images:
            img_path = train_dir / dataset_name / img_name
            if img_path.exists():
                img = cv2.imread(str(img_path))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ax = plt.subplot(n_scenes_to_show + 1, n_samples_per_scene, plot_idx) # +1 for outliers row
                ax.imshow(img)
                
                # Add decorative frame with scene color
                for spine in ax.spines.values():
                    spine.set_linewidth(3)
                    spine.set_color(scene_colors[i])
                    
                ax.set_title(f"Scene: {scene}\n{img_name}", fontsize=9, fontweight='bold', color='#333333')
                ax.axis('off')
            plot_idx += 1
        # Fill remaining plots in the row if fewer samples found
        plot_idx = ( (plot_idx -1) // n_samples_per_scene + 1) * n_samples_per_scene + 1


    # Show samples from outliers if they exist
    if 'outliers' in scenes:
         outlier_images = dataset_labels[dataset_labels['scene'] == 'outliers']['image'].sample(min(n_samples_per_scene, len(dataset_labels[dataset_labels['scene'] == 'outliers']))).tolist()
         # Adjust starting plot index for outlier row
         plot_idx = n_scenes_to_show * n_samples_per_scene + 1
         color_idx = len(regular_scenes) % len(scene_colors)  # Get next color for outliers
         
         for img_name in outlier_images:
             img_path = train_dir / dataset_name / 'outliers' / img_name # Check specific outlier folder if structure dictates
             # Fallback to main folder if outlier folder doesn't exist or image isn't there
             if not img_path.exists():
                 img_path = train_dir / dataset_name / img_name

             if img_path.exists():
                 img = cv2.imread(str(img_path))
                 img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                 ax = plt.subplot(n_scenes_to_show + 1, n_samples_per_scene, plot_idx)
                 ax.imshow(img)
                 
                 # Add decorative frame with distinctive color for outliers
                 for spine in ax.spines.values():
                    spine.set_linewidth(3)
                    spine.set_color('crimson')  # Distinctive color for outliers
                 
                 ax.set_title(f"Scene: outliers\n{img_name}", fontsize=9, fontweight='bold', color='#333333')
                 ax.axis('off')
             plot_idx += 1

    # Add dataset information header
    plt.suptitle(f"Sample Images from Dataset: {dataset_name}", fontsize=18, fontweight='bold', color='#222222')
    plt.figtext(0.5, 0.01, f"Total images in {dataset_name}: {len(dataset_labels)}", ha='center', 
                fontsize=12, bbox={"facecolor":"#e0e0e0", "alpha":0.8, "pad":5, "boxstyle":"round"})
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
    plt.show()

def visualize_dataset_statistics(dataframe, palette_name="tab10"):
    """
    Create comprehensive visualizations for dataset statistics.
    
    Parameters:
    -----------
    dataframe : pandas.DataFrame
        DataFrame containing at least 'dataset' and 'scene' columns
    palette_name : str, default 'tab10'
        Name of the seaborn color palette to use
        
    Returns:
    --------
    dict
        Dictionary containing calculated statistics:
        - scenes_per_dataset: Count of scenes in each dataset
        - images_per_scene: DataFrame with image counts per scene
        - outliers_per_dataset: Count of outlier images per dataset
        - images_per_dataset: Total image counts per dataset
    """
    # Set a clean style and define a color palette
    sns.set(style='whitegrid')
    palette = sns.color_palette(palette_name)

    # Calculate scenes per dataset
    scenes_per_dataset = dataframe.groupby('dataset')['scene'].nunique()

    # Calculate images per scene (excluding outliers for distribution stats)
    images_per_scene = (
        dataframe[dataframe['scene'] != 'outliers']
        .groupby(['dataset', 'scene'])
        .size()
        .reset_index(name='image_count')
    )

    # Outlier counts per dataset
    outliers_per_dataset = dataframe[dataframe['scene'] == 'outliers'].groupby('dataset').size()

    # Total images per dataset
    images_per_dataset = dataframe.groupby('dataset').size()

    # Print stats
    print("\n--- Scenes per Dataset (including 'outliers' as a scene) ---")
    print(scenes_per_dataset)
    print("\n--- Statistics on Images per Scene (excluding outliers) ---")
    print(images_per_scene['image_count'].describe())
    print("\n--- Outlier Images per Dataset ---")
    print(outliers_per_dataset)
    print("\n--- Total Images per Dataset ---")
    print(images_per_dataset)

    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1) Scenes per dataset
    sns.barplot(
        x=scenes_per_dataset.index,
        y=scenes_per_dataset.values,
        palette=palette,
        ax=axes[0, 0]
    )
    axes[0, 0].set_title('Number of Scenes per Dataset')
    axes[0, 0].set_xlabel('Dataset')
    axes[0, 0].set_ylabel('Number of Scenes')
    axes[0, 0].tick_params(axis='x', rotation=45)

    # 2) Distribution of images per scene
    sns.histplot(
        images_per_scene['image_count'],
        bins=30,
        kde=True,
        color=palette[2],
        ax=axes[0, 1]
    )
    axes[0, 1].set_title('Distribution of Images per Scene\n(Excluding Outliers)')
    axes[0, 1].set_xlabel('Images per Scene')
    axes[0, 1].set_ylabel('Frequency')

    # 3) Outlier images per dataset
    if not outliers_per_dataset.empty:
        sns.barplot(
            x=outliers_per_dataset.index,
            y=outliers_per_dataset.values,
            palette=palette,
            ax=axes[1, 0]
        )
        axes[1, 0].set_title('Number of Outlier Images per Dataset')
        axes[1, 0].set_xlabel('Dataset')
        axes[1, 0].set_ylabel('Outlier Count')
        axes[1, 0].tick_params(axis='x', rotation=45)
    else:
        axes[1, 0].text(
            0.5, 0.5, 'No Outliers Found',
            ha='center', va='center', fontsize=14
        )
        axes[1, 0].set_title('Outlier Images per Dataset')
        axes[1, 0].axis('off')

    # 4) Total images per dataset
    sns.barplot(
        x=images_per_dataset.index,
        y=images_per_dataset.values,
        palette=palette,
        ax=axes[1, 1]
    )
    axes[1, 1].set_title('Total Number of Images per Dataset')
    axes[1, 1].set_xlabel('Dataset')
    axes[1, 1].set_ylabel('Image Count')
    axes[1, 1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()
    
    # Return dictionary of statistics for further analysis
    return {
        'scenes_per_dataset': scenes_per_dataset,
        'images_per_scene': images_per_scene,
        'outliers_per_dataset': outliers_per_dataset,
        'images_per_dataset': images_per_dataset
    }

def create_dataset_scene_sunburst(dataframe, color_schemes=None, title="Dataset and Scene Hierarchy Visualization"):
    """
    Create an interactive sunburst visualization showing the hierarchical relationship
    between datasets and scenes.
    
    Parameters:
    -----------
    dataframe : pandas.DataFrame
        DataFrame containing at least 'dataset' and 'scene' columns
    color_schemes : list or None, default None
        List of plotly color schemes to use (e.g. px.colors.qualitative.Bold)
        If None, uses a default combination of Bold, Pastel, and Vivid
    title : str, default "Dataset and Scene Hierarchy Visualization"
        Title for the visualization
        
    Returns:
    --------
    plotly.graph_objects.Figure
        The interactive sunburst figure
    """
    import plotly.express as px
    
    # Prepare data
    sunburst_df = dataframe.groupby(['dataset', 'scene']).size().reset_index(name='count')
    
    # Build a rich color sequence if not provided
    if color_schemes is None:
        color_sequence = (
            px.colors.qualitative.Bold
            + px.colors.qualitative.Pastel
            + px.colors.qualitative.Vivid
        )
    else:
        color_sequence = []
        for scheme in color_schemes:
            color_sequence.extend(scheme)
    
    # Create sunburst
    fig = px.sunburst(
        sunburst_df,
        path=['dataset', 'scene'],
        values='count',
        color='dataset',
        color_discrete_sequence=color_sequence,
        hover_data=['count'],
        title=title
    )
    
    # Richer hover info
    fig.update_traces(
        hovertemplate=
            "<b>%{label}</b><br>"
            "Images: %{value}<br>"
            "Percent of parent: %{percentParent:.1%}<br>"
            "Percent of root: %{percentRoot:.1%}<extra></extra>"
    )
    
    # Layout tweaks
    fig.update_layout(
        title_x=0.5,
        margin=dict(t=80, l=0, r=0, b=60),
        paper_bgcolor='#f8f9fa',
        plot_bgcolor='#f8f9fa',
        font=dict(family='Arial', size=14),
        hoverlabel=dict(bgcolor='white', font_size=14),
    )
    
    # Footnote annotation
    fig.add_annotation(
        text=(
            f"Total datasets: {sunburst_df['dataset'].nunique()} | "
            f"Total scenes: {sunburst_df['scene'].nunique()} | "
            f"Total images: {sunburst_df['count'].sum()}"
        ),
        x=0.5, y=-0.05, xref='paper', yref='paper',
        showarrow=False,
        font=dict(size=14, color='#555'),
        bgcolor='#f0f0f0',
        bordercolor='#d0d0d0',
        borderwidth=1,
        borderpad=4,
        opacity=0.8
    )
    
    return fig
