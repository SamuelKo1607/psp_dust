remove(list=ls())
setwd(dir = "C:\\Users\\skoci\\Documents\\psp_dust")
myfile_solo = 'data_synced\\solo_flux_readable.csv' 
myfile_psp = 'data_synced\\psp_flux_readable.csv'

#install.packages("INLA",repos=c(getOption("repos"),INLA="https://inla.r-inla-download.org/R/stable"), dep=TRUE)
#install.packages("distr")

library(INLA)
library(hexbin)
require(hexbin)
require(lattice)
library(distr)
#require(RColorBrewer)
#library(vioplot)



###################################
### DEFINE MODEL with R-generic ###
###################################

two_component_model <- function(cmd = c("graph", "Q", "mu", "initial", 
                                          "log.norm.const", "log.prior", "quit",
                                          "rate", 
                                          "prior.l_a",
                                          "prior.l_b",
                                          "prior.v_b_r",
                                          "prior.e_v",
                                          "prior.e_b_r",
                                          "prior.shield_miss_rate"), 
                                  theta=NULL, feed_x=NULL, feed_c=NULL, feed_h=NULL){

  envir <- parent.env(environment())
  prec.high = exp(15)
  
  prior.l_a <- function(l_a=feed_x){ #around 1e-4
    return(dgamma(l_a,   shape = 5,    scale = 2e-5, log=TRUE))
  }
  prior.l_b <- function(l_b=feed_x){ #around 1e-4
    return(dgamma(l_b,   shape = 5,    scale = 2e-5, log=TRUE))
  }
  prior.v_b_r <- function(v_b_r=feed_x){ #around 50
    #return(dnorm(v_b_r,  mean  = 63.4,    sd   = 6.7, log=TRUE))
    return(dnorm(v_b_r,  mean  = 50.0,    sd   = 5, log=TRUE))
  }
  prior.e_v <- function(e_v=feed_x){ #around 1
    #return(dnorm(e_v,    mean  = 2.04,   sd   = 0.2, log=TRUE))
    return(dgamma(e_v,   shape = 5,    scale = 2e-1, log=TRUE))
  }
  prior.e_b_r <- function(e_b_r=feed_x){ #around 0.5
    #return(dnorm(e_b_r,    mean  = -1.61, sd   = 0.16, log=TRUE))
    return(dbeta(e_b_r,  shape1 = 4,  shape2 = 4, log=TRUE))
  }
  prior.shield_miss_rate <- function(shield_miss_rate=feed_x){ #around 0.5
    #return(dnorm(shield_miss_rate,  mean  = 0.5, sd   = 0.2, log=TRUE))
    return(dbeta(shield_miss_rate,  shape1 = 4,  shape2 = 4, log=TRUE))
  }
  
  rate <- function(#covariates
                   v_sc_r = feed_c[1], 
                   v_sc_t = feed_c[2], 
                   r_sc =   feed_c[3], 
                   area_front =  feed_c[4],
                   area_side =   feed_c[5],
                   heat_shield = feed_c[6],
                   #hyperparam
                   l_a =              feed_h[1], 
                   l_b =              feed_h[2], 
                   v_b_r =            feed_h[3], 
                   e_v =              feed_h[4], 
                   e_b_r =            feed_h[5],
                   shield_miss_rate = feed_h[6],
                   #bound dust parameters
                   e_a_r = -1.3,
                   #beta met. parameters
                   v_b_a = 9,
                   #stationary Earth as a normalization for the dust flux
                   v_earth_a = 0){            
    
    deg2rad <- function(deg){
      rad = deg/180*pi
      return(rad)
    }

    #beta meteoroid contribution
    ksi = -2 - (-1.5-e_b_r)
    r_factor = r_sc/1
    v_factor = ( (
        ( v_sc_r - ( v_b_r*(r_factor^ksi)  ) )^2 
        + ( v_sc_t - ( v_b_a*(r_factor^(-1)) ) )^2
      )^0.5 
      ) / ( (
        ( v_b_r )^2 
        + ( v_earth_a - v_b_a )^2
      )^0.5 
      )
    radial_impact_velocity = -1* ( v_sc_r - ( v_b_r*(r_factor^ksi) ) ) 
      #positive is on the heatshield, negative on the tail
    azimuthal_impact_velocity = abs( v_sc_t - ( v_b_a*(r_factor^(-1)) ) )
      #always positive, RHS vs LHS plays no role
    impact_angle = atan( azimuthal_impact_velocity / radial_impact_velocity )
    
    frontside = radial_impact_velocity > 0
    backside = (frontside != TRUE)
    area = (   frontside   * (1 - shield_miss_rate * heat_shield) 
                  * area_front * cos(impact_angle) 
               + backside  * 1 
                  * area_front * cos(impact_angle) 
               + area_side * sin(abs(impact_angle)) )
    
    L_b = l_b * area * (v_factor)^(e_v+1) * (r_factor)^(-1.5-e_b_r)
    
    #bound dust contribution
    ksi = -2 - e_a_r
    r_factor = r_sc/1
    v_a_a = 29.8*(r_factor^(-0.5))  
    v_factor = ( (
        ( v_sc_r )^2 
        + ( v_sc_t - v_a_a )^2
      )^0.5 
      ) / abs( v_earth_a - v_a_a )
    radial_impact_velocity = -1* ( v_sc_r ) 
      #positive is on the heatshield, negative on the tail
    azimuthal_impact_velocity = abs( v_sc_t - v_a_a )
      #always positive, RHS vs LHS plays no role
    impact_angle = atan( azimuthal_impact_velocity / radial_impact_velocity )
    
    frontside = radial_impact_velocity > 0
    backside = (frontside != TRUE)
    area = (   frontside   * (1 - shield_miss_rate * heat_shield) 
                  * area_front * cos(impact_angle) 
               + backside  * 1 
                  * area_front * cos(impact_angle) 
               + area_side * sin(abs(impact_angle)) )
    
    L_a = l_a * area * (v_factor)^(e_v+1) * (r_factor)^e_a_r
    
    #normalization to hourly rate, while L_i are in s^-1
    hourly_rate = 3600 * ( L_b + L_a )
    return(hourly_rate)
  }
  
  
  
  interpret.theta <- function(){
    return(list(l_a   =            exp(theta[1L]),
                l_b   =            exp(theta[2L]),
                v_b_r =            theta[3L],
                e_v   =            exp(theta[4L]), 
                e_b_r =            exp(theta[5L]),
                shield_miss_rate = exp(theta[6L])
               ))
  }
  
  graph <-function(){
    G <- Diagonal(n = length(vt), x=1)
    return(G)
  }
  
  Q <- function(){
    #prec.high <- interpret.theta()$prec
    Q <- prec.high*graph()
    return(Q)
  }
  
  mu <- function(){
    par = interpret.theta()
    return(log( rate(#covariates
                     vr, vt, r, area_front, area_side, heat_shield,
                     #hyperparameters
                     par$l_a, 
                     par$l_b, 
                     par$v_b_r, 
                     par$e_v, 
                     par$e_b_r,
                     par$shield_miss_rate
                     )
              ))
  }
  
  log.norm.const <-function(){
    return(numeric(0))
  }
  
  # Log-prior for thetas
  log.prior <- function(){
    par = interpret.theta()
    
    #nice priors
    val <- (prior.l_a(              par$l_a)     + theta[1L] +
            prior.l_b(              par$l_b)     + theta[2L] +
            prior.v_b_r(            par$v_b_r)   + 
            prior.e_v(              par$e_v)     + theta[4L] +
            prior.e_b_r(            par$e_b_r)   + theta[5L] +
            prior.shield_miss_rate( par$shield_miss_rate) + theta[6L]
           )
  
    return(val)
  }
  
  # Initial values of theta
  initial <- function(){
    #initial values set to the maxima a priori
    return(c(log(optimize(prior.l_a,     interval = c(0, 1e-2),       maximum = TRUE, tol=1e-9)$maximum),
             log(optimize(prior.l_b,     interval = c(0, 1e-2),       maximum = TRUE, tol=1e-9)$maximum),
                 optimize(prior.v_b_r,   interval = c(0, 1000),       maximum = TRUE, tol=1e-6)$maximum,
             log(optimize(prior.e_v,     interval = c(0, 100),        maximum = TRUE, tol=1e-6)$maximum),
             log(optimize(prior.e_b_r,   interval = c(0, 1),          maximum = TRUE, tol=1e-6)$maximum),
             log(optimize(prior.shield_miss_rate, interval = c(0, 1), maximum = TRUE, tol=1e-6)$maximum)
             )
          )
  }
  
  quit <-function(){
    return(invisible())
  }
  
  val <- do.call(match.arg(cmd), args = list())
  return(val)
}



###################################
########## Load the data ##########
###################################

mydata_solo = read.csv(file = myfile_solo)
names(mydata_solo)[c(2,3,4,5,6,9,10,11)] = c("flux",
                                             "vr",
                                             "vt",
                                             "r",
                                             "exposure",
                                             "vx",
                                             "vy",
                                             "vz")
mydata_solo$dev_angle = 0
mydata_solo$area_front = 10.34
mydata_solo$area_side = 8.24
mydata_solo$sc_id = 1
mydata_solo$heat_shield = 0

mydata_psp = read.csv(file = myfile_psp)
names(mydata_psp)[c(2,3,4,5,6,9,10,11,12,13,14)] = c("flux",
                                                  "vr",
                                                  "vt",
                                                  "r",
                                                  "exposure",
                                                  "vx",
                                                  "vy",
                                                  "vz",
                                                  "dev_angle",
                                                  "area_front",
                                                  "area_side")
mydata_psp$sc_id = 2
mydata_psp$heat_shield = 1

mydata_solo_replicated = do.call("rbind",replicate(1, mydata_solo, simplify = FALSE))
#mydata_solo_replicated = do.call("rbind",replicate(4, mydata_solo, simplify = FALSE))
#mydata_solo_replicated = do.call("rbind",replicate(20, mydata_solo, simplify = FALSE))


mydata = rbind(mydata_solo_replicated,mydata_psp)



#filterinng to far-from the Sun only
mydata <- subset(mydata, r > 0.4 & exposure > 1e-6)
n = length(mydata$vr)
mydata$idx = 1:n


###################################
########## Run the model ##########
###################################


rgen = inla.rgeneric.define(model = two_component_model, 
                            vr = mydata$vr, 
                            vt = mydata$vt, 
                            r  = mydata$r,
                            area_front = mydata$area_front,
                            area_side  = mydata$area_side,
                            heat_shield = mydata$heat_shield)
result = inla(flux ~ -1 + f(idx, model = rgen),
              data = mydata, family = "poisson", E = exposure, 
              control.compute = list(cpo=TRUE, dic=TRUE, config = TRUE),
              control.inla=list(
              control.vb=list(enable=FALSE) 
              ,int.strategy="eb"  #int.strategy="grid"
              #,strategy="adaptive" # ,strategy="gaussian"
              ),
              safe = TRUE,
              verbose = TRUE)


summary(result)

par(mfrow = c(1, 1))
hist(result$cpo$pit)     # ok 
max(result$cpo$failure)       # also OK
#pit = result$cpo$pit
#save(pit, file = "998_generated\\inla\\pit.RData")

#plotting histogram
span = round(max(abs(mydata$flux/mydata$exposure-result$summary.fitted.values$mean),40)+0.5)
hist(mydata$flux/mydata$exposure-result$summary.fitted.values$mean,
     breaks=c(-span:span),
     main="")
mtext(paste("residuals histogram, stdev = ",
            as.character(sqrt(var(mydata$flux/mydata$exposure-result$summary.fitted.values$mean))),
            ", log(mlik) = ",
            result$mlik[1]), side=3)
  
# Posterior means of the hyperparameters
l_a.mean =   inla.emarginal(function(x) exp(x), result$marginals.hyperpar$`Theta1 for idx`)
l_b.mean =   inla.emarginal(function(x) exp(x), result$marginals.hyperpar$`Theta2 for idx`)
v_b_r.mean =                                    result$summary.hyperpar$mean[3]
e_v.mean =   inla.emarginal(function(x) exp(x), result$marginals.hyperpar$`Theta4 for idx`)
e_b_r.mean = inla.emarginal(function(x) exp(x), result$marginals.hyperpar$`Theta5 for idx`)
shield_miss_rate.mean = inla.emarginal(function(x) exp(x), result$marginals.hyperpar$`Theta6 for idx`)

# Create a layout with one column and six rows
par(mfrow = c(6, 1), mar = c(2, 2, 1, 1))
# Plot each function in a separate row
plot( exp(result$marginals.hyperpar$`Theta1 for idx`[1:43]),result$marginals.hyperpar$`Theta1 for idx`[44:86])
text(x = par("usr")[1] + 0.05 * diff(par("usr")[1:2]), y = par("usr")[4] - 0.3 * diff(par("usr")[3:4]), labels = "l_a", col = "red", cex = 1.5)
plot( exp(result$marginals.hyperpar$`Theta2 for idx`[1:43]),result$marginals.hyperpar$`Theta2 for idx`[44:86])
text(x = par("usr")[1] + 0.05 * diff(par("usr")[1:2]), y = par("usr")[4] - 0.3 * diff(par("usr")[3:4]), labels = "l_b", col = "red", cex = 1.5)
plot(    (result$marginals.hyperpar$`Theta3 for idx`[1:43]),result$marginals.hyperpar$`Theta3 for idx`[44:86])
text(x = par("usr")[1] + 0.05 * diff(par("usr")[1:2]), y = par("usr")[4] - 0.3 * diff(par("usr")[3:4]), labels = "v_b_r", col = "red", cex = 1.5)
plot( exp(result$marginals.hyperpar$`Theta4 for idx`[1:43]),result$marginals.hyperpar$`Theta4 for idx`[44:86])
text(x = par("usr")[1] + 0.05 * diff(par("usr")[1:2]), y = par("usr")[4] - 0.3 * diff(par("usr")[3:4]), labels = "e_v", col = "red", cex = 1.5)
plot( exp(result$marginals.hyperpar$`Theta5 for idx`[1:43]),result$marginals.hyperpar$`Theta5 for idx`[44:86])
text(x = par("usr")[1] + 0.05 * diff(par("usr")[1:2]), y = par("usr")[4] - 0.3 * diff(par("usr")[3:4]), labels = "e_b_r", col = "red", cex = 1.5)
plot( exp(result$marginals.hyperpar$`Theta6 for idx`[1:43]),result$marginals.hyperpar$`Theta6 for idx`[44:86])
text(x = par("usr")[1] + 0.05 * diff(par("usr")[1:2]), y = par("usr")[4] - 0.3 * diff(par("usr")[3:4]), labels = "shield_miss_rate", col = "red", cex = 1.5)
# Reset the layout to the default (1x1)
par(mfrow = c(1, 1))


###################################
####### Rate decomposition ########
###################################

len = length(mydata$Julian.date)
total_rate = numeric(len)
bound_rate = numeric(len)
beta_rate = numeric(len)

samples = 100
s = inla.hyperpar.sample(samples, result)
sample_l_a   =            exp(s[,1])
sample_l_b   =            exp(s[,2])
sample_v_b_r =                s[,3]
sample_e_v =              exp(s[,4])
sample_e_b_r =            exp(s[,5])
sample_shield_miss_rate = exp(s[,6])

for (i in 1:len) {
  particuler_total_rates = numeric(samples)
  particuler_bound_rates = numeric(samples)
  particuler_beta_rates  = numeric(samples)
  for (j in 1:samples) {
    particuler_total_rates[j] <- two_component_model(cmd="rate", 
                                           feed_c = c(mydata$vr[i],
                                                      mydata$vt[i],
                                                      mydata$r[i],
                                                      mydata$area_front[i],
                                                      mydata$area_side[i],
                                                      mydata$heat_shield[i]),
                                           feed_h = c(sample_l_a[j],
                                                      sample_l_b[j],
                                                      sample_v_b_r[j],
                                                      sample_e_v[j],
                                                      sample_e_b_r[j],
                                                      sample_shield_miss_rate[j]
                                                      ))
    particuler_bound_rates[j] <- two_component_model(cmd="rate", 
                                           feed_c = c(mydata$vr[i],
                                                      mydata$vt[i],
                                                      mydata$r[i],
                                                      mydata$area_front[i],
                                                      mydata$area_side[i],
                                                      mydata$heat_shield[i]),
                                           feed_h = c(sample_l_a[j],
                                                      0,
                                                      sample_v_b_r[j],
                                                      sample_e_v[j],
                                                      sample_e_b_r[j],
                                                      sample_shield_miss_rate[j]
                                                      ))
    particuler_beta_rates[j] <- two_component_model(cmd="rate", 
                                           feed_c = c(mydata$vr[i],
                                                      mydata$vt[i],
                                                      mydata$r[i],
                                                      mydata$area_front[i],
                                                      mydata$area_side[i],
                                                      mydata$heat_shield[i]),
                                           feed_h = c(0,
                                                      sample_l_b[j],
                                                      sample_v_b_r[j],
                                                      sample_e_v[j],
                                                      sample_e_b_r[j],
                                                      sample_shield_miss_rate[j]
                                                      ))

  }
  total_rate[i] <- mean(particuler_total_rates)
  bound_rate[i] <- mean(particuler_bound_rates)
  beta_rate[i]  <- mean(particuler_beta_rates)
}

par(mfrow = c(1, 1))
plot(mydata$flux/mydata$exposure, ylab="counts/E")
lines(result$summary.fitted.values$mean, col=2, lwd=3)
#normalization due to random effect, has no meaning with a single spacecraft
lines(bound_rate / total_rate * result$summary.fitted.values$mean, col=3, lwd=3)
lines(beta_rate / total_rate * result$summary.fitted.values$mean, col=4, lwd=3)
legend(0, 50, legend=c("total", "bound", "beta"),
       col = c(2, 3, 4),
       lty = c(1, 1, 1),
       lwd = c(3, 3, 3))




###################################
### Priors and post. evaluation ###
###################################

#posteriors to be saved in X/Y form
#l_a
fx_l_a =   exp(result$marginals.hyperpar$`Theta1 for idx`[1:43])
fy_l_a =       result$marginals.hyperpar$`Theta1 for idx`[44:86]
#l_b
fx_l_b =   exp(result$marginals.hyperpar$`Theta2 for idx`[1:43])
fy_l_b =       result$marginals.hyperpar$`Theta2 for idx`[44:86]
#v_b_r
fx_v_b_r =     result$marginals.hyperpar$`Theta3 for idx`[1:43]
fy_v_b_r =     result$marginals.hyperpar$`Theta3 for idx`[44:86]
#e_v
fx_e_v =   exp(result$marginals.hyperpar$`Theta4 for idx`[1:43])
fy_e_v =       result$marginals.hyperpar$`Theta4 for idx`[44:86]
#e_b_r
fx_e_b_r = exp(result$marginals.hyperpar$`Theta5 for idx`[1:43])
fy_e_b_r =     result$marginals.hyperpar$`Theta5 for idx`[44:86]
#shield_miss_rate
fx_shield_miss_rate = exp(result$marginals.hyperpar$`Theta6 for idx`[1:43])
fy_shield_miss_rate =     result$marginals.hyperpar$`Theta6 for idx`[44:86]

#priors to be saved in X/Y form

#log priors extracted
prior.l_a                <- function(x){
  return((two_component_model(cmd="prior.l_a",               feed_x=x))) }
prior.l_b                <- function(x){
  return((two_component_model(cmd="prior.l_b",               feed_x=x))) }
prior.v_b_r              <- function(x){
  return((two_component_model(cmd="prior.v_b_r",             feed_x=x))) }
prior.e_v                <- function(x){
  return((two_component_model(cmd="prior.e_v",               feed_x=x))) }
prior.e_b_r              <- function(x){
  return((two_component_model(cmd="prior.e_b_r",             feed_x=x))) }
prior.shield_miss_rate   <- function(x){
  return((two_component_model(cmd="prior.shield_miss_rate",  feed_x=x))) }

#priors evaluated
min_x_prior <- function(x_span_posterior,margin=5) {
  result <- min(x_span_posterior)-margin*(max(x_span_posterior)
                                          -min(x_span_posterior)) }
max_x_prior <- function(x_span_posterior,margin=5) {
  result <- max(x_span_posterior)+margin*(max(x_span_posterior)
                                          -min(x_span_posterior)) }

#l_a
px_l_a = seq(0, max_x_prior(fx_l_a)*5, length.out = 100000)
py_l_a = exp(prior.l_a(px_l_a))

#l_b
px_l_b = seq(0, max_x_prior(fx_l_b)*5, length.out = 100000)
py_l_b = exp(prior.l_b(px_l_b))

#v_b_r
px_v_b_r = seq(0, 
               200, length.out = 100000)
py_v_b_r = exp(prior.v_b_r(px_v_b_r))

#e_v
px_e_v = seq(0, 
             10, length.out = 100000)
py_e_v = exp(prior.e_v(px_e_v))

#e_b_r
px_e_b_r = seq(0, 
               1, length.out = 100000)
py_e_b_r = exp(prior.e_b_r(px_e_b_r))

#shield_miss_rate
px_shield_miss_rate = seq(0, 
                          1, length.out = 100000)
py_shield_miss_rate = exp(prior.shield_miss_rate(px_shield_miss_rate))




###################################
######## Sample posterior  ########
###################################

s = inla.hyperpar.sample(1000000, result)

sample_l_a   =            exp(s[,1])
sample_l_b   =            exp(s[,2])
sample_v_b_r =            s[,3]
sample_e_v =              exp(s[,4])
sample_e_b_r =            exp(s[,5])
sample_shield_miss_rate = exp(s[,6])


#hexbinplot(sample_v_b_r~sample_l_a, 
#           data=data.frame(sample_v_b_r,sample_l_a), 
#           colramp=colorRampPalette(c("grey", "yellow")),
#           main="joint histogram v_b_r, l_a" ,  
#           xlab="l_a", 
#           ylab="v_b_r" ,
#           panel=function(x, y, ...)
#           {
#             panel.hexbinplot(x, y, ...)
#             panel.abline(v=c(mean(sample_l_a)), h=c(mean(sample_v_b_r)), 
#                          col="black", lwd=2, lty=3)
#           }
#)


###################################
#### Save everything relevant #####
###################################

model_definition <- deparse(two_component_model)

current_time <- Sys.time()
formatted_time <- format(current_time, "%Y%m%d%H%M%S")

save(sample_l_a,        #sampled posterior
     sample_l_b, 
     sample_v_b_r, 
     sample_e_v,
     sample_e_b_r,
     sample_shield_miss_rate,
     fx_l_a,          #evaluated posterior
     fy_l_a,
     fx_l_b,
     fy_l_b,
     fx_v_b_r,
     fy_v_b_r,
     fx_e_v,
     fy_e_v,
     fx_e_b_r,
     fy_e_b_r,
     fx_shield_miss_rate,
     fy_shield_miss_rate,
     px_l_a,            #evaluated prior
     py_l_a,
     px_l_b,
     py_l_b,
     px_v_b_r,
     py_v_b_r,
     px_e_v,
     py_e_v,
     px_e_b_r,
     py_e_b_r,
     px_shield_miss_rate,
     py_shield_miss_rate,
     model_definition,  #the model definition
     mydata,
     file = paste("998_generated\\inla\\solo_psp_together_sample_",
                  formatted_time,".RData",
                  sep = ""))






