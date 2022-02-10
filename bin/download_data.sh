
# Необходимо выбрать один из следующих датасетов:
# facades - f2l
# horse2zebra - h2z
# watermelons2pumpkins - w2p
# mini horse2zebra - mini


echo "Choosen dataset:"
if [ "$1" == "w2p" ]; then
echo 'download from goodle drive'
elif [ "$1" == "h2z" ]; then
wget https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/horse2zebra.zip
unzip horse2zebra.zip -d ../
rm horse2zebra.zip
elif [ "$1" == "f2l" ]; then
wget https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/facades.zip
unzip facades.zip -d ../
rm facades.zip
elif [ "$1" == "mini" ]; then
wget https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/mini.zip
unzip mini.zip -d ../
rm mini.zip
fi
