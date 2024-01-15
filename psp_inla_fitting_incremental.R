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
                                        "prior.shield_sens"), 
                                  theta=NULL, feed_x=NULL, feed_c=NULL, feed_h=NULL){

  envir <- parent.env(environment())
  prec.high = exp(15)
  
  prior.l_a <- function(l_a=feed_x){
    return(dgamma(l_a,   shape = 2,    scale = 1e-4, log=TRUE))
  }
  prior.shield_sens <- function(shield_sens=feed_x){
    return(dbeta(shield_sens,  shape1 = 2,  shape2 = 2, log=TRUE))
  }
  
  rate <- function(#covariates
                   v_sc_r = feed_c[1], 
                   v_sc_t = feed_c[2], 
                   r_sc =   feed_c[3], 
                   v_sc_x = feed_c[4], 
                   v_sc_y = feed_c[5], 
                   v_sc_z = feed_c[6], 
                   area_front = feed_c[7],
                   area_side =  feed_c[8],
                   #hyperparam
                   l_a =         feed_h[1], 
                   shield_sens = feed_h[2],
                   #bound dust parameters
                   e_a_v=2.04,
                   e_a_r=-1.3,
                   #beta met. parameters
                   v_b_a=9,
                   v_b_r=63.4,
                   e_b_v=2.04,
                   e_b_r=-1.61,
                   l_b=1.96,
                   l_bg=0,
                   #stationary Earth
                   v_earth_a=0){            
    
    deg2rad <- function(deg){
      rad = deg/180*pi
      return(rad)
    }

    #beta meteoroid contribution, old SolO model
    L_b_raw = ( (((v_sc_r-v_b_r)**2+
                  (v_sc_t-(v_b_a/r_sc))**2)**0.5)
                /50 )**(e_b_v)*r_sc**(e_b_r)*l_b + l_bg
    
    radial_impact_velocity = -1*(v_sc_r-v_b_r) 
      #positive is on the heatshield, negative on the tail
    azimuthal_impact_velocity = abs(v_sc_t-(v_b_a/r_sc))
      #always positive, RHS vs LHS plays no role
    impact_angle = atan( azimuthal_impact_velocity / radial_impact_velocity )
    
    frontside = radial_impact_velocity > 0
    backside = (frontside != TRUE)
    area = (   frontside * shield_sens * area_front * cos(impact_angle) 
               + backside  * 1           * area_front * cos(impact_angle) 
               + area_side * sin(abs(impact_angle)) )
    
    L_b = L_b_raw * area / 3600 # to get to [sec^-1]
    
    #bound dust contribution
    ksi = -2 - e_a_r
    r_factor = r_sc/1
    v_a_a = 29.8*(r_factor^(-1))
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
    area = (   frontside * shield_sens * area_front * cos(impact_angle) 
             + backside  * 1           * area_front * cos(impact_angle) 
             + area_side * sin(abs(impact_angle)) )
    
    L_a = l_a * area * (v_factor)^e_a_v * (r_factor)^e_a_r
    
    #normalization to hourly rate, while L_i are in s^-1
    hourly_rate = 3600 * ( L_b + L_a )
    return(hourly_rate)
  }
  
  
  
  interpret.theta <- function(){
    return(list(l_a   = exp(theta[1L]),
                shield_sens = exp(theta[2L])
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
                     vr, vt, r, vx, vy, vz, area_front, area_side,
                     #hyperparameters
                     par$l_a, 
                     par$shield_sens
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
    val <- (prior.l_a(          par$l_a)         + theta[1L] +
            prior.shield_sens(  par$shield_sens) + theta[2L]
           )
  
    return(val)
  }
  
  # Initial values of theta
  initial <- function(){
    #initial values set to the maxima a priori
    return(c(log(optimize(prior.l_a,  interval = c(0, 1e-2), maximum = TRUE, tol=1e-9)$maximum),
             log(optimize(prior.shield_sens, interval = c(0, 1), maximum = TRUE, tol=1e-6)$maximum)
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
mydata_solo$area_front = 10
mydata_solo$area_side = 8
mydata_solo$sc_id = 1

mydata_psp = read.csv(file = myfile_psp)
names(mydata_psp)[c(2,3,4,5,6,9,10,11,12,13)] = c("flux",
                                                  "vr",
                                                  "vt",
                                                  "r",
                                                  "exposure",
                                                  "vx",
                                                  "vy",
                                                  "vz",
                                                  "area_front",
                                                  "area_side")
mydata_psp$sc_id = 2

mydata = rbind(mydata_solo,mydata_psp)

n = length(mydata$vr)
mydata$idx = 1:n 

#filterinng to far-from the Sun only
mydata_far <- subset(mydata, r > 0.5 & sc_id == 2 & exposure > 1e-6)


###################################
########## Run the model ##########
###################################


rgen = inla.rgeneric.define(model = two_component_model, 
                            vr = mydata$vr, 
                            vt = mydata$vt, 
                            r  = mydata$r,
                            vx = mydata$vx,
                            vy = mydata$vy,
                            vz = mydata$vz,
                            area_front = mydata$area_front,
                            area_side  = mydata$area_side)
result = inla(flux ~ -1 + f(idx, model = rgen),
              data = mydata_far, family = "poisson", E = exposure, 
              control.compute = list(cpo=TRUE, dic=TRUE, config = TRUE),
              safe = TRUE, verbose = TRUE)

summary(result)

par(mfrow = c(1, 1))
hist(result$cpo$pit)     # ok 
max(result$cpo$failure)       # also OK
#pit = result$cpo$pit
#save(pit, file = "998_generated\\inla\\pit.RData")

#plotting histogram
span = round(max(abs(mydata_far$flux/mydata_far$exposure-result$summary.fitted.values$mean),40)+0.5)
hist(mydata_far$flux/mydata_far$exposure-result$summary.fitted.values$mean,
     breaks=c(-span:span),
     main="")
mtext(paste("residuals histogram, stdev = ",
            as.character(sqrt(var(mydata_far$flux/mydata_far$exposure-result$summary.fitted.values$mean))),
            ", log(mlik) = ",
            result$mlik[1]), side=3)
  
# Posterior means of the hyperparameters
l_a.mean = inla.emarginal(function(x) exp(x), result$marginals.hyperpar$`Theta1 for idx`)
shield_sens.mean = inla.emarginal(function(x) exp(x), result$marginals.hyperpar$`Theta2 for idx`)

# Create a layout with one column and seven rows
par(mfrow = c(2, 1), mar = c(2, 2, 1, 1))
# Plot each function in a separate row
plot(exp(result$marginals.hyperpar$`Theta1 for idx`[1:43]),result$marginals.hyperpar$`Theta1 for idx`[44:86])
text(x = par("usr")[1] + 0.05 * diff(par("usr")[1:2]), y = par("usr")[4] - 0.3 * diff(par("usr")[3:4]), labels = "l_a", col = "red", cex = 1.5)
plot(exp(result$marginals.hyperpar$`Theta2 for idx`[1:43]),result$marginals.hyperpar$`Theta2 for idx`[44:86])
text(x = par("usr")[1] + 0.05 * diff(par("usr")[1:2]), y = par("usr")[4] - 0.3 * diff(par("usr")[3:4]), labels = "shield_sens", col = "red", cex = 1.5)
# Reset the layout to the default (1x1)
par(mfrow = c(1, 1))


###################################
####### Rate decomposition ########
###################################

len = length(mydata_far$Julian.date)
total_rate = numeric(len)
bound_rate = numeric(len)
beta_rate = numeric(len)

samples = 100
s = inla.hyperpar.sample(samples, result)
sample_l_a   =       exp(s[,1])
sample_shield_sens = exp(s[,2])

for (i in 1:len) {
  particuler_total_rates = numeric(samples)
  particuler_bound_rates = numeric(samples)
  particuler_beta_rates  = numeric(samples)
  for (j in 1:samples) {
    particuler_total_rates[j] <- two_component_model(cmd="rate", 
                                           feed_c = c(mydata_far$vr[i],
                                                      mydata_far$vt[i],
                                                      mydata_far$r[i],
                                                      mydata_far$vx[i],
                                                      mydata_far$vy[i],
                                                      mydata_far$vz[i],
                                                      mydata_far$area_front[i],
                                                      mydata_far$area_side[i]),
                                           feed_h = c(sample_l_a[j],
                                                      sample_shield_sens[j]))
    particuler_beta_rates[j] <- two_component_model(cmd="rate", 
                                           feed_c = c(mydata_far$vr[i],
                                                      mydata_far$vt[i],
                                                      mydata_far$r[i],
                                                      mydata_far$vx[i],
                                                      mydata_far$vy[i],
                                                      mydata_far$vz[i],
                                                      mydata_far$area_front[i],
                                                      mydata_far$area_side[i]),
                                           feed_h = c(0,
                                                      sample_shield_sens[j]))

  }
  total_rate[i] <- mean(particuler_total_rates)
  beta_rate[i]  <- mean(particuler_beta_rates)
}

par(mfrow = c(1, 1))
plot(mydata_far$flux/mydata_far$exposure, ylab="counts/E")
lines(result$summary.fitted.values$mean, col=2, lwd=3)
#normalization due to random effect, has no meaning with a single spacecraft
lines(beta_rate / total_rate * result$summary.fitted.values$mean, col=4, lwd=3)
legend(0, 50, legend=c("total", "beta"),
       col = c(2, 4),
       lty = c(1, 1),
       lwd = c(3, 3))




###################################
### Priors and post. evaluation ###
###################################

#posteriors to be saved in X/Y form
#l_a
fx_l_a = exp(result$marginals.hyperpar$`Theta1 for idx`[1:43])
fy_l_a = result$marginals.hyperpar$`Theta1 for idx`[44:86]
#l_b
fx_l_b = exp(result$marginals.hyperpar$`Theta2 for idx`[1:43])
fy_l_b = result$marginals.hyperpar$`Theta2 for idx`[44:86]
#v_b_r
fx_v_b_r = result$marginals.hyperpar$`Theta3 for idx`[1:43]
fy_v_b_r = result$marginals.hyperpar$`Theta3 for idx`[44:86]
#e_a_v
fx_e_a_v = result$marginals.hyperpar$`Theta4 for idx`[1:43]
fy_e_a_v = result$marginals.hyperpar$`Theta4 for idx`[44:86]
#e_b_v
fx_e_b_v = result$marginals.hyperpar$`Theta5 for idx`[1:43]
fy_e_b_v = result$marginals.hyperpar$`Theta5 for idx`[44:86]
#e_a_r
fx_e_a_r = result$marginals.hyperpar$`Theta6 for idx`[1:43]
fy_e_a_r = result$marginals.hyperpar$`Theta6 for idx`[44:86]
#e_b_r
fx_e_b_r = result$marginals.hyperpar$`Theta7 for idx`[1:43]
fy_e_b_r = result$marginals.hyperpar$`Theta7 for idx`[44:86]
#shield_sens
fx_shield_sens = exp(result$marginals.hyperpar$`Theta8 for idx`[1:43])
fy_shield_sens = result$marginals.hyperpar$`Theta8 for idx`[44:86]

#priors to be saved in X/Y form

#log priors extracted
prior.l_a <- function(x){
  return((three_component_model(cmd="prior.l_a",    feed_x=x))) }
prior.l_b   <- function(x){
  return((three_component_model(cmd="prior.l_b",    feed_x=x))) }
prior.v_b_r <- function(x){
  return((three_component_model(cmd="prior.v_b_r",  feed_x=x))) }
prior.e_a_v   <- function(x){
  return((three_component_model(cmd="prior.e_a_v",  feed_x=x))) }
prior.e_b_v   <- function(x){
  return((three_component_model(cmd="prior.e_b_v",  feed_x=x))) }
prior.e_a_r   <- function(x){
  return((three_component_model(cmd="prior.e_a_r",  feed_x=x))) }
prior.e_b_r   <- function(x){
  return((three_component_model(cmd="prior.e_b_r",  feed_x=x))) }
prior.shield_sens   <- function(x){
  return((three_component_model(cmd="prior.shield_sens",  feed_x=x))) }

#priors evaluated
#l_a
min_x_prior <- function(x_span_posterior,margin=5) {
  result <- min(x_span_posterior)-margin*(max(x_span_posterior)-min(x_span_posterior)) }
max_x_prior <- function(x_span_posterior,margin=5) {
  result <- max(x_span_posterior)+margin*(max(x_span_posterior)-min(x_span_posterior)) }

px_l_a = seq(0, max_x_prior(fx_l_a)*5, length.out = 100000)
py_l_a = exp(prior.l_a(px_l_a))
#l_b

px_l_b = seq(0, max_x_prior(fx_l_b)*5, length.out = 100000)
py_l_b = exp(prior.l_b(px_l_b))
#v_b_r

px_v_b_r = seq(min_x_prior(fx_v_b_r), max_x_prior(fx_v_b_r), length.out = 100000)
py_v_b_r = exp(prior.v_b_r(px_v_b_r))
#e_a_v

px_e_a_v = seq(min_x_prior(fx_e_a_v), max_x_prior(fx_e_a_v), length.out = 100000)
py_e_a_v = exp(prior.e_a_v(px_e_a_v))
#e_b_v

px_e_b_v = seq(min_x_prior(fx_e_b_v), max_x_prior(fx_e_b_v), length.out = 100000)
py_e_b_v = exp(prior.e_b_v(px_e_b_v))
#e_a_r

px_e_a_r = seq(min_x_prior(fx_e_a_r), max_x_prior(fx_e_a_r), length.out = 100000)
py_e_a_r = exp(prior.e_a_r(px_e_a_r))
#e_b_r

px_e_b_r = seq(min_x_prior(fx_e_b_r), max_x_prior(fx_e_b_r), length.out = 100000)
py_e_b_r = exp(prior.e_b_r(px_e_b_r))
#shield_sens

px_shield_sens = seq(0, 1, length.out = 10000)
py_shield_sens = exp(prior.shield_sens(px_shield_sens))

###################################
######## Sample posterior  ########
###################################

s = inla.hyperpar.sample(1000000, result)

sample_l_a   =       exp(s[,1])
sample_l_b   =       exp(s[,2])
sample_v_b_r =           s[,3]
sample_e_a_v =           s[,4]
sample_e_b_v =           s[,5]
sample_e_a_r =           s[,6]
sample_e_b_r =           s[,7]
sample_shield_sens = exp(s[,8])

hexbinplot(sample_v_b_r~sample_l_a, 
           data=data.frame(sample_v_b_r,sample_l_a), 
           colramp=colorRampPalette(c("grey", "yellow")),
           main="joint histogram v_b_r, l_a" ,  
           xlab="l_a", 
           ylab="v_b_r" ,
           panel=function(x, y, ...)
           {
             panel.hexbinplot(x, y, ...)
             panel.abline(v=c(mean(sample_l_a)), h=c(mean(sample_v_b_r)), col="black", lwd=2, lty=3)
           }
)


###################################
#### Save everything relevant #####
###################################

model_definition <- deparse(three_component_model)

current_time <- Sys.time()
formatted_time <- format(current_time, "%Y%m%d%H%M%S")

save(sample_l_a,        #sampled posterior
     sample_l_b, 
     sample_v_b_r, 
     sample_e_a_v,
     sample_e_b_v,
     sample_e_a_r,
     sample_e_b_r,
     sample_shield_sens,
     fx_l_a,          #evaluated posterior
     fy_l_a,
     fx_l_b,
     fy_l_b,
     fx_v_b_r,
     fy_v_b_r,
     fx_e_a_v,
     fy_e_a_v,
     fx_e_b_v,
     fy_e_b_v,
     fx_e_a_r,
     fy_e_a_r,
     fx_e_b_r,
     fy_e_b_r,
     fx_shield_sens,
     fy_shield_sens,
     px_l_a,            #evaluated prior
     py_l_a,
     px_l_b,
     py_l_b,
     px_v_b_r,
     py_v_b_r,
     px_e_a_v,
     py_e_a_v,
     px_e_b_v,
     py_e_b_v,
     px_e_a_r,
     py_e_a_r,
     px_e_b_r,
     py_e_b_r,
     px_shield_sens,
     py_shield_sens,
     model_definition,  #the model definition
     mydata,
     file = paste("998_generated\\inla\\sample_",formatted_time,".RData",
                  sep = ""))






