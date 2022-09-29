import wolf

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
    
    resources = {"cpus-per-task": 4, "mem" : "6G"}
    docker = "gcr.io/broad-getzlab-workflows/facets:v0"
