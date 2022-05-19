library(ggplot2)
library(grid)
library(gridExtra)
library(ggpubr)
library(ggfortify)
library(cluster)
library(clusterCrit)
library(funtimes)

# GRID SEARCH
pam.gridSearch <- function(data, repeats, maxClusters) {
    
    numClustersCol = c()
    metricCol = c()
    standarizationCol = c()
    purityCol = c()
    silhouetteCol = c()
    
    for (clusters in 2:maxClusters) {
        for (stand in c(TRUE, FALSE)) {
            for (metric in c("euclidean", "manhattan")) {
                purityValues = c()
                silhouetteValues = c()
                
                for (i in 1:repeats) {

                    # Set seed for reproducibility between params
                    set.seed(i)

                    # Fit and compute purity and silhouette
                    pam.fit <- pam(data[,1:length(data)-1], k=clusters, stand=stand, metric=metric)
                    pam.purity <- purity(data$class, pam.fit$cluster)
                    pam.silhouette <- silhouette(pam.fit$cluster, dist((data[,1:length(data)-1])))
                    
                    # Collect data
                    purityValues = c(purityValues, pam.purity$pur)
                    silhouetteValues = c(mean(pam.silhouette[, 3]))  
                }
                
                # Fill entry
                numClustersCol <- c(numClustersCol, clusters)
                metricCol <- c(metricCol, metric)
                standarizationCol <- c(standarizationCol, stand)
                purityCol <- c(purityCol, mean(purityValues))
                silhouetteCol <- c(silhouetteCol, mean(silhouetteValues))   
            }
        }
    }
    
    df <- data.frame(
        num_clusters = unlist(numClustersCol),
        metric = unlist(metricCol),
        standarization = as.factor(unlist(standarizationCol)),
        Purity = unlist(purityCol),
        Silhouette = unlist(silhouetteCol)
    )
    return (df)
}

pam.gridSearch.vis <- function(results) {
    g1 <- ggplot(df, aes(x=num_clusters, y=Purity, color=standarization)) + 
        geom_line(size = 1.0) + geom_point(size = 3.0) + facet_wrap(~metric) +
        theme(text = element_text(size = 18)) + rremove("xlab")
    
    g2 <- ggplot(df, aes(x=num_clusters, y=Silhouette, color=standarization)) + 
        geom_line(size = 1.0) + geom_point(size = 3.0) + facet_wrap(~metric) +
        theme(text = element_text(size = 18)) + 
        theme(strip.background = element_blank(), strip.text.x = element_blank())
    
    G <- grid.arrange(
        g1, g2,
        nrow = 2,
        top = textGrob("PAM: grid search", gp=gpar(fontsize=22))
    )
    
    return (G)
}
