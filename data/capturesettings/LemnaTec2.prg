-- Program Start -- |
New Record|
Measure|On
Set Meas. Light = |3,0,20,1,=
Set Meas. Freq. = |3,0,3,3
Set Act. Light = |5,0,20,0,=
Set Act. Width = |0,0,900,0,=
Set Gain = |5,1,20,2,=
Set Damping = |2,0,5,2,=
Set Sat. Light = |10,1,10,10,=
Spacer| 
TimeStep(s) =|10,1,32000,2
FvFm|
TimeStep(s) =|10,1,32000,40
Set Act. Light = |5,0,20,8,=
TimeStep(s) =|10,1,32000,1
Yield|
Begin of Repetition Block |induction curve
TimeStep(s) =|10,1,32000,20
Yield|
End of Repetition Block; Loops = |2,1,32000,13
Set Act. Light = |5,0,20,0,0
Measure|Off
