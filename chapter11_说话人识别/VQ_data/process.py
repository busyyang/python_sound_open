from sphfile import SPHFile
import glob

if __name__ == "__main__":
    path = r'*.WAV'
    sph_files = glob.glob(path)
    print(len(sph_files))
    for i in sph_files:
        print(i)
        sph = SPHFile(i)
        filename = i.replace(".WAV", ".wav")
        sph.write_wav(filename)

    print("Completed")
