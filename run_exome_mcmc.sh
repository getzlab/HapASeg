#!/bin/bash
for CLUST in {0..19}
do
	hapaseg --output_dir ./exome_results/ coverage_mcmc --coverage_csv ./exome/6_C1D1_META.cov --allelic_clusters_object ./exome/6_C1D1_META.DP_clusts.auto_ref_correct.overdispersion92.no_phase_correct.npz --SNPs_pickle ./exome/6_C1D1_META.SNPs.pickle --covariate_dir ./covars --num_draws 10 --cluster_num $CLUST --allelic_sample 499
done
