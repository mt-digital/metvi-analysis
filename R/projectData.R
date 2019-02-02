project.data <- function(data.path, pre.elec, twenty.sixteen)
{
    df <- read.csv(data.path)
    
    if (pre.elec)
    {
        elec.date <- ifelse(twenty.sixteen, '2016-11-8', '2012-11-6')
        
        df <- df[ as.Date(df$date) < as.Date(elec.date), ]
    }
    # print(df$date)
    # The formulae have an index column for fitting overdispersion.
    df$index <- 1:nrow(df)
    
    return (df)
}
