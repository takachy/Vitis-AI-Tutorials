#**************************************************************************************
# Copyright 2021 Xilinx Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#*******************************************************************************/

#### just for info ####
set tool "unknown"
set version "unknown"

foreach toolcheck {vivado_hls vitis_hls} {
    if {[ string first $toolcheck $argv0 ]>=0} {
        set tool $toolcheck
        set version "[exec which $toolcheck | sed -e "s:.*\\(20..\\..\\).*:\\1:g" ]"
    }
}

puts "*** tool: $tool is version: $version ***"
####

set do_cosim 0

open_project vhls_dpupreproc_prj

# using variable mytop to set which version we're using:
# [] hls_dpupreproc is the reference model from Dan and uses AXIS interfaces
# [] hls_dpupreproc_m_axi is derived from the above but using the M_AXI interfaces
# commenting/uncommenting will change the solution name; note this is only driven from the script not possible from the GUI
#set mytop hls_dpupreproc
set mytop hls_dpupreproc_m_axi

set_top $mytop

set mycflags "-I ./src -std=c++14"

add_files     src/dpupreproc_vhls.cpp -cflags $mycflags
add_files -tb src/dpupreproc_ref.cpp  -cflags $mycflags
add_files -tb src/dpupreproc_tb.cpp   -cflags $mycflags
add_files -tb src/dpupreproc_main.cpp -cflags $mycflags
add_files -tb src/ap_bmp.cpp          -cflags $mycflags
add_files -tb data_pre

#open_solution "solution_$mytop" -flow_target vivado
open_solution "solution_$mytop" -flow_target vitis
set_part {xcvc1902-vsva2197-2MP-e}
create_clock -period 3 -name default

config_interface -m_axi_alignment_byte_size 64 -m_axi_latency 64 -m_axi_max_widen_bitwidth 512 -m_axi_offset slave
config_rtl -register_reset_num 3


csim_design -clean
csynth_design
cosim_design
#export_design -flow impl -rtl verilog -format xo
















csim_design  -clean

csynth_design

if {$do_cosim > 0} {
    cosim_design
}

##export_design -flow syn -rtl verilog -format ip_catalog
#export_design -rtl verilog -format xo -output ./$mytop.xo

exit
