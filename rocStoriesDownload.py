import urllib.request
import os


def download_dataset(url, filename):
    urllib.request.urlretrieve(url, os.path.join(os.getcwd(), filename))


if __name__ == '__main__':
    datasets = {
        "ROCStories_winter_2017": "https://goo.gl/0OYkPK",
        "ROCStories_spring_2016": "https://goo.gl/7R59b1",
        "Story_Cloze_Test_Winter_2018_val": "https://goo.gl/XWjas1",
        "Story_Cloze_Test_Winter_2018_test": "https://goo.gl/BcTtB4",
        "Story_Cloze_Test_Spring_2016_val": "https://goo.gl/cDmS6I",
        "Story_Cloze_Test_Spring_2016_test": "https://goo.gl/iE31Qm"
    }

    for name, url in datasets.items():
        download_dataset(url, f"{name}.csv")
        print(f"Downloaded {name}.csv")
