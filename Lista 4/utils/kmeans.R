library(ggplot2)
library(grid)
library(gridExtra)
library(ggpubr)
library(ggfortify)
library(cluster)
library(clusterCrit)
library(funtimes)

# MAS ITERS
kmeans.maxIters.search <- function(data, repeats, maxIters) {
    options(warn=-1)
    
    clusters = length(unique(data$class))
    
    maxItersCol = c()
    preprocessingCol = c()
    purityCol = c()
    silhouetteCol = c()
    
    for (iters in 1:maxIters) {
        for (preprocessing in c("none", "standarization", "normalization")) {
            purityValues = c()
            silhouetteValues = c()

            for (i in 1:repeats) {

                # Set seed for reproducibility between params
                set.seed(i)

                # Preprocessing
                features <- data[,1:length(data)-1]
                if (preprocessing == "standarization") {
                    features <- sapply(features, FUN=function(x) (x - mean(x)) / (sd(x)))
                } else if (preprocessing == "normalization") {
                    features <- sapply(features, FUN=function(x) (x - min(x)) / (max(x) - min(x)))
                }

                # Fit and compute purity and silhouette
                kmeans.fit <- kmeans(features, centers=clusters, iter.max=iters)
                kmeans.purity <- purity(data$class, kmeans.fit$cluster)
                kmeans.silhouette <- silhouette(kmeans.fit$cluster, dist(features))

                # Collect data
                purityValues = c(purityValues, kmeans.purity$pur)
                silhouetteValues = c(mean(kmeans.silhouette[, 3])) 
            }

            # Fill entry
            maxItersCol <- c(maxItersCol, iters)
            preprocessingCol <- c(preprocessingCol, preprocessing)
            purityCol <- c(purityCol, mean(purityValues))
            silhouetteCol <- c(silhouetteCol, mean(silhouetteValues))   
        }
    }
    
    df <- data.frame(
        max_iters = unlist(maxItersCol),
        preprocessing = as.factor(unlist(preprocessingCol)),
        Purity = unlist(purityCol),
        Silhouette = unlist(silhouetteCol)
    )
    
    options(warn=0)
    return (df)
}

kmeans.maxIters.vis <- function(results) {
    g1 <- ggplot(df, aes(x=max_iters, y=Purity, color=preprocessing)) + 
        geom_line(size = 1.0) + geom_point(size = 3.0) +
        theme(text = element_text(size = 18)) + rremove("xlab")
    
    g2 <- ggplot(df, aes(x=max_iters, y=Silhouette, color=preprocessing)) + 
        geom_line(size = 1.0) + geom_point(size = 3.0) +
        theme(text = element_text(size = 18)) + 
        theme(strip.background = element_blank(), strip.text.x = element_blank())
    
    G <- grid.arrange(
        g1, g2,
        nrow = 2,
        top = textGrob("K-means: max iterations analysis", gp=gpar(fontsize=22))
    )
    
    return (G)
}

# GRID SEARCH
kmeans.gridSearch <- function(data, repeats, maxClusters, nstartArray) {
    
    numClustersCol = c()
    nstartCol = c()
    preprocessingCol = c()
    purityCol = c()
    silhouetteCol = c()
    
    for (clusters in 2:maxClusters) {
        for (preprocessing in c("none", "standarization", "normalization")) {
            for (nstart in nstartArray) {
                purityValues = c()
                silhouetteValues = c()
                
                for (i in 1:repeats) {

                    # Set seed for reproducibility between params
                    set.seed(i)
                    
                    # Preprocessing
                    features <- data[,1:length(data)-1]
                    if (preprocessing == "standarization") {
                        features <- sapply(features, FUN=function(x) (x - mean(x)) / (sd(x)))
                    } else if (preprocessing == "normalization") {
                        features <- sapply(features, FUN=function(x) (x - min(x)) / (max(x) - min(x)))
                    }

                    # Fit and compute purity and silhouette
                    kmeans.fit <- kmeans(features, centers=clusters, nstart=nstart)
                    kmeans.purity <- purity(data$class, kmeans.fit$cluster)
                    kmeans.silhouette <- silhouette(kmeans.fit$cluster, dist(features))
                    
                    # Collect data
                    purityValues = c(purityValues, kmeans.purity$pur)
                    silhouetteValues = c(mean(kmeans.silhouette[, 3])) 
                }
                
                # Fill entry
                numClustersCol <- c(numClustersCol, clusters)
                nstartCol <- c(nstartCol, nstart)
                preprocessingCol <- c(preprocessingCol, preprocessing)
                purityCol <- c(purityCol, mean(purityValues))
                silhouetteCol <- c(silhouetteCol, mean(silhouetteValues))   
            }
        }
    }
    
    df <- data.frame(
        num_clusters = unlist(numClustersCol),
        nstart = as.factor(unlist(nstartCol)),
        preprocessing = as.factor(unlist(preprocessingCol)),
        Purity = unlist(purityCol),
        Silhouette = unlist(silhouetteCol)
    )
    return (df)
}

kmeans.gridSearch.vis <- function(results) {
    g1 <- ggplot(df, aes(x=num_clusters, y=Purity, color=nstart)) + 
        geom_line(size = 1.0) + geom_point(size = 3.0) + facet_wrap(~preprocessing) +
        theme(text = element_text(size = 18)) + rremove("xlab")
    
    g2 <- ggplot(df, aes(x=num_clusters, y=Silhouette, color=nstart)) + 
        geom_line(size = 1.0) + geom_point(size = 3.0) + facet_wrap(~preprocessing) +
        theme(text = element_text(size = 18)) + 
        theme(strip.background = element_blank(), strip.text.x = element_blank())
    
    G <- grid.arrange(
        g1, g2,
        nrow = 2,
        top = textGrob("K-means: grid search", gp=gpar(fontsize=22))
    )
    
    return (G)
}
