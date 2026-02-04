# source /workspace/setup_env.sh



python -m analyses.plot_histogram --activations-dir outputs/transcripts/upgrade/allenai_Olmo-3.1-32B-Instruct/main/activations/geodesic-research_sfm_unfiltered_e2e_misalignment_upsampled_base/main/ --filtering-dir outputs/transcripts/upgrade/allenai_Olmo-3.1-32B-Instruct/main/filtering/ --minimum-rating 2

python -m analyses.plot_histogram --activations-dir outputs/transcripts/upgrade/allenai_Olmo-3.1-32B-Instruct/main/activations/geodesic-research_sfm_unfiltered_e2e_alignment_upsampled_base/main/ --filtering-dir outputs/transcripts/upgrade/allenai_Olmo-3.1-32B-Instruct/main/filtering/ --minimum-rating 2

python -m analyses.plot_histogram --activations-dir outputs/transcripts/upgrade/allenai_Olmo-3.1-32B-Instruct/main/activations/geodesic-research_sfm_filtered_e2e_alignment_upsampled_base/main/ --filtering-dir outputs/transcripts/upgrade/allenai_Olmo-3.1-32B-Instruct/main/filtering/ --minimum-rating 2

python -m analyses.plot_histogram --activations-dir outputs/transcripts/upgrade/allenai_Olmo-3.1-32B-Instruct/main/activations/geodesic-research_sfm_baseline_filtered_base/main/ --filtering-dir outputs/transcripts/upgrade/allenai_Olmo-3.1-32B-Instruct/main/filtering/ --minimum-rating 2

python -m analyses.plot_histogram --activations-dir outputs/transcripts/upgrade/allenai_Olmo-3.1-32B-Instruct/main/activations/geodesic-research_sfm_baseline_unfiltered_base/main/ --filtering-dir outputs/transcripts/upgrade/allenai_Olmo-3.1-32B-Instruct/main/filtering/ --minimum-rating 2

python -m analyses.plot_histogram --activations-dir outputs/transcripts/upgrade/allenai_Olmo-3.1-32B-Instruct/main/activations/geodesic-research_sfm_unfiltered_cpt_misalignment_upsampled_base/main/ --filtering-dir outputs/transcripts/upgrade/allenai_Olmo-3.1-32B-Instruct/main/filtering/ --minimum-rating 2

python -m analyses.plot_histogram --activations-dir outputs/transcripts/upgrade/allenai_Olmo-3.1-32B-Instruct/main/activations/geodesic-research_sfm_unfiltered_cpt_alignment_upsampled_base/main/ --filtering-dir outputs/transcripts/upgrade/allenai_Olmo-3.1-32B-Instruct/main/filtering/ --minimum-rating 2

python -m analyses.plot_histogram --activations-dir outputs/transcripts/upgrade/allenai_Olmo-3.1-32B-Instruct/main/activations/geodesic-research_sfm_filtered_cpt_alignment_upsampled_base/main/ --filtering-dir outputs/transcripts/upgrade/allenai_Olmo-3.1-32B-Instruct/main/filtering/ --minimum-rating 2




