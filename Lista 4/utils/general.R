library(ggplot2)
library(grid)
library(gridExtra)
library(ggpubr)
library(ggfortify)
library(cluster)
library(clusterCrit)
library(funtimes)

# DEFAULT CLUSTERING COMPARISON
clustering.vis.samples <- function(data, samples) {
    clusters = length(unique(data$class))
    plots = list()
    
    for (i in 1:samples) {
        set.seed(i)
        
        kmeans.fit <- kmeans(data[,1:length(data)-1], centers=clusters)
        pam.fit <- pam(data[,1:length(data)-1], k=clusters)
        
        g1 <- autoplot(kmeans.fit, data=data, colour="class", shape="class", frame=TRUE, frame.type='norm')
        g2 <- autoplot(kmeans.fit, data=data, frame=TRUE, shape="class", frame.type='norm')
        g3 <- autoplot(pam.fit, data=data, frame=TRUE, shape="class", frame.type='norm')
        
        if (i == 1) {
            g1 <- g1 + labs(title="Dataset classes")
            g2 <- g2 + labs(title="K-means clusters")
            g3 <- g3 + labs(title="PAM clusters")
        }
        
        if (i != samples) {
            g1 <- g1 + rremove("xlab")
            g2 <- g2 + rremove("xlab")
            g3 <- g3 + rremove("xlab")
        }
        
        new_plots <- list(
            g1 + theme(text = element_text(size = 14)),
            g2 + theme(text = element_text(size = 14)) + rremove("ylab"),
            g3 + theme(text = element_text(size = 14)) + rremove("ylab")
        )
        plots <- append(plots, new_plots)
    }
    
    G <- grid.arrange(
        grobs = plots,
        ncol = 3,
        nrow = samples,
        top = textGrob("Clustering samples (random initial centroids)", gp=gpar(fontsize=22))
    )
    
    return (G)
}
