#!/usr/bin/env python

import time
import random
import argparse

from pixivpy3 import AppPixivAPI, PixivError

REFRESH_TOKEN_FILE = "refresh_token.secret"

SLEEP_MIN = 1
SLEEP_MAX = 2


def remove_newpage_tags(text):
    # Remove all occurrences of '[newpage]'
    cleaned_text = text.replace('[newpage]', '')
    return cleaned_text


def get_novel(aapi, novel_id):
    json_result = aapi.novel_detail(novel_id)
    novel = json_result.novel

    json_result = aapi.novel_text(novel_id)

    return novel.title + "\n\n" + json_result.novel_text


def get_series(aapi, series_id):
    texts = []
    json_result = aapi.novel_series(series_id)
    for novel in json_result.novels:
        texts.append(get_novel(aapi, novel.id))
        time.sleep(random.uniform(SLEEP_MIN, SLEEP_MAX))

    # get next page
    next_qs = aapi.parse_qs(json_result.next_url)
    while next_qs is not None:
        json_result = aapi.novel_series(**next_qs)
        for novel in json_result.no1360613vels:
            texts.append(get_novel(aapi, novel.id))
            time.sleep(random.uniform(SLEEP_MIN, SLEEP_MAX))
        next_qs = aapi.parse_qs(json_result.next_url)

    return remove_newpage_tags("\n\n\n".join(texts))


def main():
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description='Download novel or series from Pixiv.')
    parser.add_argument('novel_id', type=int, nargs='?', help='Novel ID to download')
    parser.add_argument('-s', '--series', type=int, help='Series ID to download')

    args = parser.parse_args()

    aapi = AppPixivAPI()

    with open(REFRESH_TOKEN_FILE, 'r') as file:
        refresh_token = file.read().strip()

    _e = None
    for _ in range(3):
        try:
            aapi.auth(refresh_token=refresh_token)
            break
        except PixivError as e:
            _e = e
            time.sleep(10)
    else:  # failed 3 times
        raise _e

    if args.series:
        # Fetch series details if series ID is provided
        with open("s{}.txt".format(args.series), "w", encoding='utf-8') as file:
            file.write(get_series(aapi, args.series))
    elif args.novel_id:
        # Fetch novel details if novel ID is provided
        with open("n{}.txt".format(args.novel_id), "w", encoding='utf-8') as file:
            file.write(get_novel(aapi, args.novel_id))
    else:
        print("No novel or series ID provided.")


if __name__ == "__main__":
    main()
