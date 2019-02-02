##
# Fit three candidate models and report the results.
#

library(rethinking)

# Avoids recompilation?
rstan_options(auto_write = TRUE)

library(plyr)

source('R/projectData.R')


# d <- project.data('full-final-forstats.csv', FALSE, TRUE)
# head(d)
d <- project.data('full-norerun-2-1-19.csv', TRUE, TRUE)

# Center days from debate to aid MCMC.
# d$days.from.c <- d$days.from.debate - mean(d$days.from.debate)
d$days.from.c <- d$daysFromDebate - mean(d$daysFromDebate)

# The Stan golem seems to need this instead of the program name.
# The program names are in order of left-to-right between networks and
# (least, most) popular of the two within network.
d$program_idx <- mapvalues(d$program_name,
                           from=c(
                             "The Last Word With Lawrence O'Donnell",
                             "The Rachel Maddow Show",
                             "Erin Burnett OutFront",
                             "Anderson Cooper 360",
                             "The Kelly File",
                             "The O'Reilly Factor"
                             ),
                           to=c(1, 2, 3, 4, 5, 6)
)

print(head(d))

# No effect from network or program, just days away from debate.
# m.days.from <- map2stan(
#     alist(
#         n ~ dpois(lambda),
#         log(lambda) <- a + bdf*days.from.c,
#         a ~ dnorm(0, 5),
#         bdf ~ dnorm(0, 5)
#     ),
#     data=d, warmup=5e3, iter=2e4, chains=6, cores=3
# )


# print('\n** Random intercepts for each program. **\n')
# m.program.intercepts <- map2stan(
#     alist(
#         n ~ dpois(lambda),
#         log(lambda) <- aa + a[program_idx],
#         a[program_idx] ~ dnorm(aa, 5),
#         aa ~ dnorm(0, 5)
#     ),
#     data=d, warmup=5e3, iter=2e4, chains=6, cores=3
# )

 
# # 
# print('\n** Varying slopes model. **\n')
# m.program.int.slopes <- map2stan(
#     alist(
#         n ~ dpois(lambda),
#         log(lambda) <- a_prog[program_idx] + bdf_prog[program_idx]*days.from.c,
#         c(a_prog, bdf_prog)[program_idx] ~ dmvnorm2(c(a, bdf), sigma_prog, Rho),
#         a ~ dnorm(0, 5),
#         bdf ~ dnorm(0, 5),
#         sigma_prog ~ dcauchy(0, 1),
#         Rho ~ dlkjcorr(5)
#     ),
#     data=d, warmup=5e3, iter=2e4, chains=6, cores=3
# )


print('\n** Program intercept and slopes, dvnorm non-centered **\n')
m.program.int.slopes.NC <- map2stan(
    alist(
        n ~ dpois(lambda),
        log(lambda) <- a_prog[program_idx] + bdf_prog[program_idx]*days.from.c,
        c(a_prog, bdf_prog)[program_idx] ~ dmvnormNC(sigma_prog, Rho_prog),
        c(a,bdf) ~ dnorm(0, 5),
        sigma_prog ~ dcauchy(0, 1),
        Rho_prog ~ dlkjcorr(4)
    ),
    data=d, warmup=5e3, iter=2e4, chains=6, cores=3
)


# print('\n** Program and network intercept and slopes, dvnorm non-centered **\n')
# m.net.program.int.slopes <- map2stan(
#     alist(
#         n ~ dpois(lambda),
#         log(lambda) <- a_prog[program_idx] + bdf_net[network]*days.from.c
#                                            + bdf_prog[program_idx]*days.from.c,
#         c(a_prog, bdf_prog)[program_idx] ~ dmvnorm2(c(a, bdf), sigma_prog, Rho_prog),
#         c(a_net, bdf_net)[network] ~ dmvnorm2(0, sigma_net, Rho_net),
#         a ~ dnorm(0, 5),
#         bdf ~ dnorm(0, 5),
#         sigma_prog ~ dcauchy(0, 1),
#         Rho_prog ~ dlkjcorr(4),
#         sigma_net ~ dcauchy(0, 1),
#         Rho_net ~ dlkjcorr(4)
#     ),
#     data=d, warmup=5e3, iter=2e4, chains=6, cores=3
# )


# print('\n** Program intercept and slopes, dvnorm non-centered **\n')
# m.net.program.int.slopes.NC <- map2stan(
#     alist(
#         n ~ dpois(lambda),
#         log(lambda) <- a_prog[program_idx] + bdf_net[network]*days.from.c
#                                            + bdf_prog[program_idx]*days.from.c,
#         c(a_prog, bdf_prog)[program_idx] ~ dmvnormNC(sigma_prog, Rho_prog),
#         c(a_net, bdf_net, Rho_net)[network] ~ dmvnormNC(sigma_net, Rho_net),
#         c(a,bdf) ~ dnorm(0, 5),
#         sigma_prog ~ dcauchy(0, 1),
#         Rho_prog ~ dlkjcorr(4),
#         sigma_net ~ dcauchy(0, 1),
#         Rho_net ~ dlkjcorr(4)
#     ),
#     data=d, warmup=5e3, iter=2e4, chains=6, cores=3
# )
