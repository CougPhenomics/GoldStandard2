#! Rscript
# Command-line script to produce timelapse videos
# Input: 
#   datadir = relative path to data-specific output directory
#   genotype_mamp = relative path to csv containing genotype info
# Example: Rscript --vanilla scripts/makeVideos.R "output/psII" "data/genotype_map.csv"

# Get command line arguments
args = commandArgs(trailingOnly = T)
# args  = c('output/psII', 'data/genotype_map.csv')

# test if there are two arguments: if not, return an error
if (length(args)<2) {
  stop('Two arguments must be supplied:\n1. output directory for a dataset\n2. path to genotype_map.csv\nFor Example: Rscript --vanilla makeVideos.R "output/psII" "diy_data/genotype_map.csv"', call.=T)
}

# function to try to load required makes. if not loadable, install.
load_install = function(pkg){
  if(!require(pkg, character.only = T)){
    install.packages(pkg, character.only =T, repos = 'https://cloud.r-project.org')
    require(pkg, character.only = T)
  }
}

# load all packages
libs = c('magick','tidyverse','lubridate','av','rprojroot')
tmp = sapply(libs, FUN = load_install)


# setup directories
root = rprojroot::find_root('.here')
datadir = args[1]
indir = file.path(root, datadir, 'pseudocolor_images')
outdir = file.path(root, datadir, 'timelapse')
dir.create(outdir, show = F, rec = T)

# get genotype info
gmap = read_csv(file.path(root,args[2])) %>% arrange(roi, plantbarcode)

# get data processed
output = read_csv(file.path(root, datadir,'output_psII_level0.csv'),
                  col_types = cols(gtype = col_character())) %>%
  dplyr::select(gtype, plantbarcode, roi)

# filter gmap for available output files
gmap = inner_join(gmap, output) %>% distinct(plantbarcode, roi, gtype) %>% 
  mutate(treatment = 'control')

# setup roi positions for genotype labels
nrow = 2
ncol = 1
nroi = nrow*ncol
rownum = floor((seq_len(nroi)-1) / nrow) + 1
colnum = floor((seq_len(nroi)-1) / ncol + 1)
x0 = 95
xoffset = 170
y0 = 100
yoffset = 260
xpos = x0 + (rownum - 1) * xoffset
ypos = y0 + (colnum - 1) * yoffset
coords = crossing(xpos, ypos) %>% arrange(ypos) %>% mutate(roi = seq_len(nroi)-1) %>% inner_join(gmap)

# function to create treatment label
get_treatment <- function(traynum) {
  paste(traynum,
        gmap %>% filter(plantbarcode == traynum) %>% distinct(treatment))
}

# get dates from filename
get_dates = function(fns) {
  splitlist = stringr::str_split(basename(fns), '[-\\ ]')
  map_chr(splitlist, .f = ~ lubridate::ymd(.x[3]) %>% as.character)
}

# create list of tray pairs for gifs
fluc_ids = unique(gmap %>% filter(treatment != 'control') %>% pull(plantbarcode))
cntrl_ids = unique(gmap %>% filter(treatment == 'control') %>% pull(plantbarcode))
if(length(fluc_ids)!=0){
  l = cross2(fluc_ids, cntrl_ids)
} else {
  l = as.list(gmap %>% distinct(plantbarcode) %>% pull)
}
# l=l[c(1,2)]

# test values
# sampleid_c = 'A1'
# sampleid_t = 'A2'
# parameter_string = 't320_ALon-NPQ'#'FvFm_YII' #
# il = l[[1]]

# define gif making function
arrange_gif = function(il, parameter_string) {
  uil = unlist(il, rec = F)
  if(length(uil)==1){
    sampleid_c = uil
  } else {
    sampleid_c = uil[2]
  }
  # Get first set of images regardless of number of samples in uil:
  # get images
  fns0 = dir(file.path(indir, sampleid_c),
             pattern = parameter_string,
             full.names = T)
  #get dates
  dtes0 = get_dates(fns0)
  
  # get genotypes
  g0 = gmap %>% filter(plantbarcode == sampleid_c) %>% pull(gtype)
  
  # read images
  imgs0 = image_read(fns0)
  
  # annotate with genotype
  imgs0a = image_annotate(
    imgs0,
    get_treatment(sampleid_c),
    size = 24,
    font = 'Arial',
    weight = 700,
    gravity = "NorthWest",
    location = geometry_point(30, 20),
    color = "white"
  )
  
  imgs0a = image_annotate(
    imgs0a,
    dtes0,
    size = 24,
    font = 'Arial',
    weight = 700,
    gravity = "NorthWest",
    location = geometry_point(300, 20),
    color = 'white'
  )
  
  coords %>%
    filter(plantbarcode == sampleid_c) %>%
    group_by(xpos, ypos, roi) %>%
    group_walk(
      keep = T,
      .f = function(df, grp) {
        # print(paste('grp:',grp, collapse=','))
        # print(str(df))
        xpos = df$xpos
        ypos = df$ypos
        gtype = df$gtype
        if (toupper(gtype) == 'WT') {
          gstyle = 'Normal'
        } else {
          gstyle = 'Italic'
        }
        # print(gtype)
        
        imgs0a <<- 
          image_annotate(
          imgs0a,
          gtype,
          font = 'Arial',
          style = gstyle,
          weight = 700,
          size = 24,
          gravity = 'NorthWest',
          location = geometry_point(xpos, ypos),
          color = 'white'
        )
      }
    )
  
  # if there is only 1 set of images save to file. otherwise get second set of images and filter for common dates, append together, write to file
  if(length(uil)==1){
    
    # combine timelapses
    for (i in 1:length(imgs0)) {
      if (i == 1) {
        newgif = imgs0a[i]
        # newgif = image_append(c(newgif,imgstif[i]), stack=T)
      } else {
        combined  <- imgs0a[i]
        # combined = image_append(c(combined,imgstif[i]), stack=T)
        newgif <- c(newgif, combined)
      }
    }
    # newgif
    outfn = paste0(parameter_string, '-', sampleid_c,'-',paste(g0, collapse='_'),'.gif')
    image_write_video(newgif, file.path(outdir, outfn), framerate = 2)
    # image_write_gif(newgif,file.path(outdir, outfn), delay=0.5)
    
  } else {
    
    sampleid_t = uil[1]
    gtype_treatment_label_t = get_treatment(sampleid_t)
    print(paste(sampleid_c, sampleid_t, sep = ' x '))
    print(gtype_treatment_label_t)
    print(parameter_string)
    
    fns1 = dir(file.path(indir, sampleid_t),
               pattern = parameter_string,
               full.names = T)
    
    # get dates from filenames
    dtes1 = get_dates(fns1)
    
    # filter for common dates
    commondtes <- intersect(dtes0,dtes1)
    elements0 = dtes0 %in% commondtes
    elements1 = dtes1 %in% commondtes
    dtes0 <- dtes0[elements0]
    dtes1 <- dtes1[elements1]
    fns0 <- fns0[elements0]
    fns1 <- fns1[elements1]
    
    # get genotypes
    g1 = gmap %>% filter(plantbarcode == sampleid_t) %>% pull(gtype)
    # crossing(dtes0,dtes1) #TODO: filter dates and filenames for common dates
    
    stopifnot(all(dtes0 == dtes1))
    
    #read images
    imgs1 = image_read(fns1)
    
    # annotate images
    
    imgs1a = image_annotate(
      imgs1,
      gtype_treatment_label_t,
      size = 24,
      font = 'Arial',
      weight = 700,
      gravity = "NorthWest",
      location = geometry_point(30, 20),
      color = "white"
    )
    imgs1a = image_annotate(
      imgs1a,
      dtes1,
      size = 24,
      font = 'Arial',
      weight = 700,
      gravity = "NorthWest",
      location = geometry_point(300, 20),
      color = 'white'
    )
    
    coords %>%
      filter(plantbarcode == sampleid_t) %>%
      group_by(xpos, ypos, roi) %>%
      group_walk(
        keep = T,
        .f = function(df, grp) {
          # print(paste('grp:',grp, collapse=','))
          # print(str(df))
          xpos = df$xpos
          ypos = df$ypos
          gtype = df$gtype
          if (toupper(gtype) == 'WT') {
            gstyle = 'Normal'
          } else {
            gstyle = 'Italic'
          }
          # #
          imgs1a <<-   image_annotate(
            imgs1a,
            gtype,
            font = 'Arial',
            style = gstyle,
            weight = 700,
            size = 24,
            gravity = 'NorthWest',
            location = geometry_point(xpos, ypos),
            color = 'white'
          )
        }
      )
    
    # combine timelapses
    for (i in 1:length(imgs0)) {
      if (i == 1) {
        newgif = image_append(c(imgs0a[i], imgs1a[i]))
        # newgif = image_append(c(newgif,imgstif[i]), stack=T)
      } else {
        combined  <- image_append(c(imgs0a[i], imgs1a[i]))
        # combined = image_append(c(combined,imgstif[i]), stack=T)
        newgif <- c(newgif, combined)
      }
    }
    # newgif
    treatmentlabel = strsplit(gtype_treatment_label_t,' ')[[1]][2]
    outfn = paste0(parameter_string, '_', sampleid_c, '_x_', sampleid_t, '_',treatmentlabel,'.gif')
    image_write_video(newgif, file.path(outdir, outfn), framerate = 2)
    # image_write_gif(newgif,file.path(outdir, outfn), delay=0.5)
  }
}

allparams = c('t80_ALon-NPQ','t320_ALon-NPQ', 'FvFm-YII', 't320_ALon-YII')
# allparams = 't80_ALon-NPQ'
for (param in allparams) {
  walk(l, arrange_gif, param)
}
