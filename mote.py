#! /usr/local/bin/python3
# -*- coding: utf-8 -*-

import math
import json
import twitter_api
import numpy as np
from sklearn.externals import joblib

MEAN = 0.5
VAR = 0.3


def extract_feature(profile):
    feature = []
    name = list(profile['name'])
    n_n_gram = gen_n_gram(name, 1)
    n_n_gram.extend(gen_n_gram(name, 2))
    n_n_gram.extend(gen_n_gram(name, 3))

    s_name = list(profile['screen_name'])[1:]
    s_n_gram = gen_n_gram(s_name, 1)
    s_n_gram.extend(gen_n_gram(s_name, 2))
    s_n_gram.extend(gen_n_gram(s_name, 3))

    desc = list(profile['description'])
    d_n_gram = gen_n_gram(desc, 1)
    d_n_gram.extend(gen_n_gram(desc, 2))
    d_n_gram.extend(gen_n_gram(desc, 3))

    loc = list(profile['location'])
    l_n_gram = gen_n_gram(loc, 1)
    l_n_gram.extend(gen_n_gram(loc, 2))
    l_n_gram.extend(gen_n_gram(loc, 3))

    cv = joblib.load('dump/name_cv.pkl')
    feature.extend(cv.transform([n_n_gram]))
    cv = joblib.load('dump/screen_cv.pkl')
    feature.extend(cv.transform([s_n_gram]))
    cv = joblib.load('dump/desc_cv.pkl')
    feature.extend(cv.transform([d_n_gram]))
    cv = joblib.load('dump/loc_cv.pkl')
    feature.extend(cv.transform([l_n_gram]))

    feature.append(int(profile['protected']))
    feature.append(np.log10(int(profile['followers_count']) + 1))
    feature.append(np.log10(int(profile['friends_count']) + 1))
    feature.append(np.log10(int(profile['statuses_count']) + 1))
    feature.append(np.log10(int(profile['media_count']) + 1))

    feature.append(
        int(profile['url'] is not None)
    )

    # if set pref language
    feature.append(
        int(profile['lang'] != 'ja')
    )

    # リストを持っているかどうか
    feature.append(
        int(profile['listed_count'] != 0)
    )

    # 色を変えているかどうか 通常1DA1F2
    feature.append(
        int(profile['profile_link_color'] != '1DA1F2')
    )

    # 色を変えているかどうか 通常333333
    feature.append(
        int(profile['profile_text_color'] != '1DA1F2')
    )

    # 色を変えているかどうか 通常C0DEED
    feature.append(
        int(profile['profile_sidebar_border_color'] != 'C0DEED')
    )

    feature.append(int(profile['contributors_enabled']))
    feature.append(int(profile['is_translator']))
    feature.append(int(profile['is_translation_enabled']))
    feature.append(int(profile['profile_use_background_image']))
    feature.append(int(profile['has_extended_profile']))
    feature.append(int(profile['default_profile']))
    feature.append(int(profile['default_profile_image']))
    feature.append(int(profile['has_custom_timelines']))
    feature.append(int(profile['can_media_tag']))
    return feature


def gen_n_gram(lst, n, delim=" "):
    return [delim.join(
                (["<s>"] * (n - 1) + lst + ["</s>"] * (n - 1))[i: i + n]
            ) for i in range(len(lst) + n - 1)]


def calc_mote(screenname):
    params = {
        "screen_name": screenname,
        "count": 200,
        "include_user_entities": True,
    }
    req = twitter_api.get_instance('followers/list', params=params)
    followers = json.loads(req.text)
    clf = joblib.load("dump/clf.pkl")

    n_male = 0
    n_female = 0
    for f in followers:
        if clf.predict(extract_feature(f)) == [1]:
            n_male += 1
        else:
            n_female += 1

    params = {
        "screen_name": screenname,
        "include_entity": True,
    }
    req = twitter_api.get_instance('users/show', params=params)
    user_profile = json.loads(req.text)
    if clf.predict(extract_feature(user_profile)) == [1]:
        rate = n_female / (n_male + n_female)
    else:
        rate = n_male / (n_male + n_female)

    return ((rate - MEAN) / math.sqrt(VAR)) * 0.1 + 0.5
