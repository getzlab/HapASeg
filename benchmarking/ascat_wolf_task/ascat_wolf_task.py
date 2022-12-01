import wolf
from wolf.localization import LocalizeToDisk

class ASCAT_Prepare_HTS(wolf.Task):
    inputs = {"normal_bam": None,
              "normal_bai": None,
              "tumor_bam": None,
              "tumor_bai": None,
              "normal_name": None,
              "tumor_name": None,
              "loci_dir":None,
              "allele_dir":None,
              "chr_notation":True, # if bams use chr contig names
              "gender": None,
              "genome_version": None,
              "num_threads":8
              }

    def script(self):
        script = """mkdir ./alleles_dir && cp ${allele_dir}/* ./alleles_dir &&\
        R -e "library(ASCAT); \
        ascat.prepareHTS('${tumor_bam}', '${normal_bam}', '${tumor_name}', '${normal_name}',\
        'alleleCounter', './alleles_dir/G1000_alleles_${genome_version}_chr', './loci_dir/G1000_loci_${genome_version}_chr', '${gender}', \
        '${genome_version}', nthreads=${num_threads})"
        """
        
        if self.conf["inputs"]["chr_notation"]:
            # rename all loci files
            chr_loci_script = """set -eo pipefail && mkdir ./loci_dir && for i in {1..22} X; do sed 's/^/chr/' ${loci_dir}/G1000_loci_${genome_version}_chr${i}.txt > ./loci_dir/G1000_loci_${genome_version}_chr${i}.txt; done && """
            script = chr_loci_script + script
        else:
            # if renaming not necessary, just move files
            mv_loci_script = """set -eo pipefail && mkdir ./loci_dir && cp ${loci_dir}/* ./loci_dir &&"""
            script = mv_loci_script + script
        return script
    
    output_patterns = {
        "tumor_BAF": "*_tumourBAF.txt", 
        "tumor_LogR": "*_tumourLogR.txt",
        "normal_BAF": "*_normalBAF.txt",
        "normal_LogR": "*_normalLogR.txt"
        }

    resources = {"cpus-per-task": 32, "mem":"180G"}

    docker = "gcr.io/broad-getzlab-workflows/ascat:v1"

def ASCAT_Generate_Raw(normal_bam = None,
                       normal_bai = None,
                       tumor_bam = None,
                       tumor_bai = None,
                       normal_name = None,
                       tumor_name = None,
                       loci_dir = None,
                       allele_dir = None,
                       chr_notation=True,
                       gender = None, #XX or XY
                       genome_version = None,
                       num_threads = 8
                       ):
    
    t_bam_localization_task = wolf.LocalizeToDisk(
        files = {
            "bam" : tumor_bam,
            "bai" : tumor_bai
                }
        )
    n_bam_localization_task = wolf.LocalizeToDisk(
        files = {
            "bam" : normal_bam,
            "bai" : normal_bai
                }
        )

    prepare_raw_task =  ASCAT_Prepare_HTS(inputs = {
              "normal_bam": n_bam_localization_task["bam"],
              "normal_bai": n_bam_localization_task["bai"],
              "tumor_bam": t_bam_localization_task["bam"],
              "tumor_bai": t_bam_localization_task["bai"],
              "normal_name": normal_name,
              "tumor_name": tumor_name,
              "loci_dir": loci_dir,
              "allele_dir": allele_dir,
              "chr_notation":chr_notation, # if bams use chr contig names
              "gender": gender,
              "genome_version": genome_version,
              "num_threads":num_threads
              },
              preemptible = False)
    
class ASCAT(wolf.Task):
    inputs = {'tumor_LogR': None,
              'tumor_BAF':None,
              'normal_LogR': None,
              'normal_BAF':None,
              'GC_correction_file': None,
              'RT_correction_file': None}
    
    script = """
    R -e "library(ASCAT); \
        ascat.bc = ascat.loadData(Tumor_LogR_file = '${tumor_LogR}',\
                                  Tumor_BAF_file = '${tumor_BAF}',\
                                  Germline_LogR_file = '${normal_LogR}', \
                                  Germline_BAF_file = '${normal_BAF}',\
                                  gender = rep('XX', 1), \
                                  genomeVersion = 'hg38'); \
        ascat.bc = ascat.correctLogR(ascat.bc, GCcontentfile = '${GC_correction_file}',\
                                     replictimingfile = '${RT_correction_file}'); \
        ascat.plotRawData(ascat.bc, img.prefix = 'After_correction_'); \
        ascat.bc = ascat.aspcf(ascat.bc); \
        ascat.output = ascat.runAscat(ascat.bc, gamma=1, write_segments = T); \
        QC = ascat.metrics(ascat.bc,ascat.output); \
        write.table(QC, './ascat_QC.txt', quote=FALSE);"
"""
    output_patterns = {
    "ascat_corrected_data_normal" : "After_correction_*.germline.png",
    "ascat_corrected_data_tumor" : "After_correction_*.tumour.png",
    "ascat_profile_plot": "*.ASCATprofile.png",
    "ascat_raw_profile" : "*.rawprofile.png",
    "ascat_raw_segments" : "*.segments_raw.txt",
    "ascat_segments" : "*.segments.txt",
    "ascat_sunrise_plot" : "*.sunrise.png"
    }
    
    resources = {"cpus-per-task": 4, "mem" : "8G"}
    docker = "gcr.io/broad-getzlab-workflows/ascat:v0"
