"""抓取两个场景的 OSM 路网、居民点与关键设施（Overpass，mail.ru 镜像）。"""
import json, subprocess, sys, time, os
URL = "https://maps.mail.ru/osm/tools/overpass/api/interpreter"
SCEN = {
    # 名称: (南, 西, 北, 东)
    "luding": (29.35, 101.85, 30.25, 102.45),   # 四川甘孜州泸定县（2022-09-05 M6.8 地震）
    "xuwen":  (20.20, 109.90, 20.62, 110.60),   # 广东湛江徐闻县（2024-09-06 台风"摩羯"登陆）
}
def q(bbox):
    s,w,n,e = bbox; b=f"({s},{w},{n},{e})"
    return f"""[out:json][timeout:300];
(
  way["highway"~"^(motorway|trunk|primary|secondary|tertiary|unclassified|residential|motorway_link|trunk_link|primary_link|secondary_link|tertiary_link|track|service)$"]{b};
);
out geom;
(
  node["place"~"^(city|town|village|hamlet|suburb|quarter|neighbourhood|isolated_dwelling|locality)$"]{b};
  node["amenity"~"^(hospital|clinic|school|townhall|police|fire_station)$"]{b};
  way["amenity"~"^(hospital|clinic|school|townhall|police|fire_station)$"]{b};
);
out center tags;
"""
for name,bbox in SCEN.items():
    out=f"data/osm_{name}.json"
    if os.path.exists(out) and os.path.getsize(out)>10000: print(name,"exists"); continue
    for k in range(4):
        r=subprocess.run(["curl","-s","-A","Mozilla/5.0 research-sim","-H","Accept: application/json","--max-time","600","-o",out,"-w","%{http_code}","-G","--data-urlencode","data="+q(bbox),URL],capture_output=True,text=True)
        code=r.stdout.strip(); sz=os.path.getsize(out) if os.path.exists(out) else 0
        print(name,"http",code,"bytes",sz)
        if code=="200" and sz>10000: break
        time.sleep(20)
    d=json.load(open(out)); els=d["elements"]
    ways=[e for e in els if e["type"]=="way" and "highway" in e.get("tags",{})]
    places=[e for e in els if e.get("tags",{}).get("place")]
    amen=[e for e in els if e.get("tags",{}).get("amenity")]
    print(name,"ways",len(ways),"places",len(places),"amenities",len(amen))
