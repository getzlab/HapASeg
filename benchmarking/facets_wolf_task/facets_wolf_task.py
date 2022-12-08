import wolf
from wolf.localization import LocalizeToDisk, UploadToBucket

class Facets_SNP_Pileup(wolf.Task):
    inputs = {"vcf": None,
              "normal_bam": None,
              "normal_bai": None,
              "tumor_bam" : None,
              "tumor_bai" : None
             }
    script = """
    snp-pileup -q15 -Q20 -P100 -r25,0 ${vcf} -p ./facets_allelecounts.txt ${normal_bam} ${tumor_bam}
    """
    output_patterns = {
    "facets_allelecounts" : "facets_allelecounts.txt",
    }
    
    resources = {"cpus-per-task": 4, "mem" : "12G"}
    docker = "gcr.io/broad-getzlab-workflows/facets:v1"

def Facets_Generate_Raw(vcf = None,
                        normal_bam=None,
                        normal_bai=None,
                        tumor_bam=None,
                        tumor_bai=None,
                        upload_bucket=None):
    
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

    pileups_task = Facets_SNP_Pileup(inputs = {"vcf": vcf,
                                "normal_bam": n_bam_localization_task["bam"],
                                "normal_bai": n_bam_localization_task["bai"],
                                "tumor_bam": t_bam_localization_task["bam"],
                                "tumor_bai": t_bam_localization_task["bai"]
                               }
                     )
    if upload_bucket is not None:
        upload_task = UploadToBucket(files = pileups_task["facets_allelecounts"],
                                     bucket = upload_bucket.rstrip("/") + '/facets/')

class Facets(wolf.Task):
    inputs = {'snp_counts': None}
    
    script = """
    R -e "library(facets); set.seed(1234); \
          rcmat=readSnpMatrix('${snp_counts}'); \
          xx=preProcSample(rcmat, gbuild='hg38', ndepth=10); \
          oo=procSample(xx, cval=150); fit=emcncf(oo); \
          write.table(fit\$cncf, './facets_segfile.txt', row.names=FALSE, quote=FALSE); \
          write.table(data.frame(purity=fit\$purity, ploidy=fit\$ploidy,  dipLogR=fit\$dipLogR), \
           './facets_out_params.txt', quote=FALSE, row.names=FALSE); \
          plotSample(x=oo, emfit=fit); dev.off(); \
          print(fit\$emflags)"
    """
    output_patterns = {
    "facets_seg_file" : "facets_segfile.txt",
    "facets_out_params": "facets_out_params.txt",
    "facets_output_plot": "Rplots.pdf"
    }
    
    resources = {"cpus-per-task": 4, "mem" : "8G"}
    docker = "gcr.io/broad-getzlab-workflows/facets:v0"
