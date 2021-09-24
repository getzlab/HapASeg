import argparse
import dask.distributed as dd
import multiprocessing

from .load import HapasegReference
from .run_allelic_MCMC import AllelicMCMCRunner

def parse_args():
    parser = argparse.ArgumentParser(description = "Call somatic copynumber alterations taking advantage of SNP phasing")

    input_group = parser.add_mutually_exclusive_group(
      required = True
      #"Mutually exclusive inputs",
      #"Hapaseg can either take an already phased VCF annotated with alt/refcounts, or a MuTect callstats file. In the case of the latter, Hapaseg will perform phasing.",
    )
    input_group.add_argument("--phased_VCF")
    input_group.add_argument("--callstats_file")

    # parser.add_argument("--coverage_file")

    # if we are taking a pre-phased VCF, we also may want a read-backed phased VCF
    prephased_group = parser.add_argument_group( 
      "Inputs if phasing has already been imputed",
    )
    prephased_group.add_argument("--read_backed_phased_VCF", help = "Optional.")
    prephased_group.add_argument("--allele_counts_T", help = "Required.")
    prephased_group.add_argument("--allele_counts_N", help = "Required.")

    # if we are performing phasing ourselves, we need these reference files
    phasing_ref_group = parser.add_argument_group(
      "Phasing ref. files",
      "Required files for Hapaseg to pull down het sites and impute phasing.",
    )
    phasing_ref_group.add_argument("--phasing_ref_panel_dir")
    phasing_ref_group.add_argument("--phasing_genetic_map_file")
    phasing_ref_group.add_argument("--SNP_list")
    phasing_ref_group.add_argument("--ref_fasta")
    phasing_ref_group.add_argument("--bam", help = "BAM to use for read-backed phasing")
    phasing_ref_group.add_argument("--bai", help = "BAI to use for read-backed phasing")

    ref_group = parser.add_argument_group(
      "Required reference files",
    )
    ref_group.add_argument("--cytoband_file", required = True)

    ai_seg_params = parser.add_argument_group(
      "Parameters for allelic imbalance segmentation",
    )
    ai_seg_params.add_argument("--n_workers", default = multiprocessing.cpu_count() - 1)
    ai_seg_params.add_argument("--n_iter", default = 20000)
    ai_seg_params.add_argument("--phase_correct", action = "store_true")
    ai_seg_params.add_argument("--misphase_prior", default = "0.001")

    args = parser.parse_args()

    # validate arguments
    if args.phased_VCF is not None:
        if args.allele_counts_T is None or args.allele_counts_N is None:
            raise ValueError("Must supply allele counts from HetPulldown if not imputing phasing")
    # TODO: if callstats_file is given, then all phasing_ref_group args must be present

    return args

def main(): 
    args = parse_args()

    dask_client = dd.Client(n_workers = args.n_workers)

    refs = HapasegReference(
      phased_VCF = read.phased_VCF,
      readbacked_phased_VCF = args.read_back_phased_VCF,
      allele_counts = args.allele_counts_T,
      allele_counts_N = args.allele_counts_N
    )

    runner = AllelicMCMCRunner(
      refs.allele_counts,
      refs.chromosome_intervals,
      dask_client,
      phase_correct = args.phase_correct
    )

    allelic_segs = runner.run_all()

if __name__ == "__main__":
    main()
