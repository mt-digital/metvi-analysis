require(arm)

df <- read.csv("full-final-forstats.csv")

fit.null <- glm (n ~ 1, family=poisson, data=df)
display(fit.null)

fit.network <- glm (n ~ network, family=poisson, data=df)
display(fit.network)

# Overdispersed.
fit.network.od <- glm (n ~ network, family=quasipoisson, data=df)
display(fit.network.od)

writeLines(c("", "", "daysFromDebate"))
fit.daysFrom <- glm (n ~ daysFromDebate, family=poisson, data=df)
display(fit.daysFrom)
#fit.log.daysFrom <- glm (n ~ log(daysFromDebate), family=poisson, data=df)

writeLines(c("", "", "daysFromDebate and network no interaction"))
fit.daysFromAndNetwork <- glm (
  n ~ daysFromDebate + network, family=poisson, data=df
)
display(fit.daysFromAndNetwork)

writeLines(c("", "", "daysFromDebate and network with interaction"))
fit.daysFromAndNetworkInt <- glm (
  n ~ daysFromDebate + network + daysFromDebate:network, family=poisson, data=df
)
display(fit.daysFromAndNetworkInt)

writeLines(c("", "", "@realDonaldTrump"))
fit.RepTweets <- glm (
  n ~ RepTweets, family=poisson, data=df
)
display(fit.RepTweets)

writeLines(c("", "", "@HillaryClinton"))
fit.DemTweets <- glm (
  n ~ DemTweets, family=poisson, data=df
)
display(fit.DemTweets)

writeLines(c("", "Both Twitter"))
fit.BothTweets <- glm(
  n ~ DemTweets + RepTweets, family=poisson, data=df
)
display(fit.BothTweets)

writeLines(c("", "Both Twitter with interaction"))
fit.BothTweetsInt <- glm(
  n ~ DemTweets + RepTweets + DemTweets:RepTweets, family=poisson, data=df
)
display(fit.BothTweetsInt)

writeLines(c("", "Both Twitter with interaction and Full Model from Before"))
fit.BothTweetsIntFull <- glm(
  n ~ DemTweets + RepTweets + DemTweets:RepTweets + daysFromDebate + 
    network + daysFromDebate:network, 
  family=poisson, data=df
)
display(fit.BothTweetsIntFull)



#######
# This doesn't seem to be working, and I think that's just as well. I think there
# are good results in the full model. I think that as more factors are 
# accounted for that cause all networks to increase the same, Fox News uses
# more and more metaphorical violence than the others.


# writeLines(c("", "Dem tweets and candidate as subject"))
# fit.DemSubjTweets <- glm(
#   DemSubj ~ DemTweets, family=poisson, data=df
# )
# display(fit.DemSubjTweets)
# 
# writeLines(c("", "Republican tweets and candidate as subject"))
# fit.RepSubjTweets <- glm(
#   RepSubj ~ RepTweets, family=poisson, data=df
# )
# display(fit.RepSubjTweets)
