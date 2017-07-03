


# IMAGE DATA
# ----------
# [offset] [type]          [value]          [description] 
# 0000     32 bit integer  0x00000803(2051) magic number 
# 0004     32 bit integer  60000            number of images 
# 0008     32 bit integer  28               number of rows 
# 0012     32 bit integer  28               number of columns 
# 0016     unsigned byte   ??               pixel 
# 0017     unsigned byte   ??               pixel 
# ........ 
# xxxx     unsigned byte   ??               pixel


# LABEL DATA
# ----------
# [offset] [type]          [value]          [description] 
# 0000     32 bit integer  0x00000801(2049) magic number (MSB first) 
# 0004     32 bit integer  60000            number of items 
# 0008     unsigned byte   ??               label 
# 0009     unsigned byte   ??               label 
# ........ 
# xxxx     unsigned byte   ??               label