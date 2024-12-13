awk '{printf("%.6f ", $1/1e9); for (i=2; i<=NF; i++) printf("%s ", $i); printf("\n");}' KeyFrameTrajectory.txt > KeyFrameTrajectory_seconds.txt
sed -i 's/[[:space:]]*$//' KeyFrameTrajectory_seconds.txt
