#ifndef SEMANTIC_DEFINITION_H
#define SEMANTIC_DEFINITION_H

#include <map>
#include <string>

#include <opencv2/opencv.hpp>

namespace SLIM {

#define NONE_FEATURE_ID   0
#define POLE_ID           80
#define ROAD_ID           40
#define SIDEWALK_ID       48
#define BUILDING_ID       50
#define FENCE_ID          51
#define TRAFFIC_SIGN_ID   81

#define POLE_RESOLUTION         0.5
#define ROAD_RESOLUTION         2.0
#define SIDEWALK_RESOLUTION     2.0
#define BUILDING_RESOLUTION     1.0
#define FENCE_RESOLUTION        1.0
#define TRAFFIC_SIGN_RESOLUTION 0.2

#define CV_COLOR_BLACK      cv::Scalar(0,0,0)          // 纯黑
#define CV_COLOR_WHITE      cv::Scalar(255,255,255)    // 纯白
#define CV_COLOR_RED        cv::Scalar(0,0,255)        // 纯红
#define CV_COLOR_GREEN      cv::Scalar(0,255,0)        // 纯绿
#define CV_COLOR_BLUE       cv::Scalar(255,0,0)        // 纯蓝

#define CV_COLOR_DARKGRAY   cv::Scalar(169,169,169)    // 深灰色
#define CV_COLOR_DARKRED    cv::Scalar(0,0,169)        // 深红色
#define CV_COLOR_ORANGERED  cv::Scalar(0,69,255)       // 橙红色

#define CV_COLOR_CHOCOLATE  cv::Scalar(30,105,210)     // 巧克力色
#define CV_COLOR_GOLD       cv::Scalar(10,215,255)     // 金色
#define CV_COLOR_YELLOW     cv::Scalar(0,255,255)      // 纯黄色

#define CV_COLOR_OLIVE      cv::Scalar(0,128,128)      // 橄榄色
#define CV_COLOR_LIGHTGREEN cv::Scalar(144,238,144)    // 浅绿色
#define CV_COLOR_DARKCYAN   cv::Scalar(139,139,0)      // 深青色
#define CV_COLOR_CYAN       cv::Scalar(255,255,0)      // 青色

#define CV_COLOR_SKYBLUE    cv::Scalar(235,206,135)    // 天蓝色 
#define CV_COLOR_INDIGO     cv::Scalar(130,0,75)       // 藏青色
#define CV_COLOR_PURPLE     cv::Scalar(128,0,128)      // 紫色

#define CV_COLOR_PINK       cv::Scalar(203,192,255)    // 粉色
#define CV_COLOR_DEEPPINK   cv::Scalar(147,20,255)     // 深粉色
#define CV_COLOR_VIOLET     cv::Scalar(238,130,238)    // 紫罗兰

static std::vector<cv::Scalar> cs_map_color = {
  CV_COLOR_DARKCYAN,
  CV_COLOR_PURPLE,
  CV_COLOR_ORANGERED,
  CV_COLOR_YELLOW,
  CV_COLOR_INDIGO,
  CV_COLOR_DEEPPINK,
  CV_COLOR_OLIVE,
  CV_COLOR_DARKRED,
  CV_COLOR_GREEN,
  CV_COLOR_BLUE
};

static std::vector<cv::Scalar> random_color_vec = {
  cv::Scalar(0, 255, 0),
  cv::Scalar(0, 0, 255),
  cv::Scalar(245, 150, 100),
  cv::Scalar(245, 230, 100),
  cv::Scalar(250, 80, 100),
  cv::Scalar(150, 60, 30),
  cv::Scalar(255, 0, 0),
  cv::Scalar(180, 30, 80),
  cv::Scalar(50, 50, 255),
  cv::Scalar(200, 40, 255),
  cv::Scalar(90, 30, 150),
  cv::Scalar(255, 0, 255),
  cv::Scalar(255, 150, 255),
  cv::Scalar(75, 0, 75),
  cv::Scalar(75, 0, 175),
  cv::Scalar(0, 200, 255),
  cv::Scalar(50, 120, 255),
  cv::Scalar(0, 150, 255),
  cv::Scalar(170, 255, 150),
  cv::Scalar(0, 175, 0),
  cv::Scalar(0, 60, 135),
  cv::Scalar(80, 240, 150),
  cv::Scalar(150, 240, 255),
  cv::Scalar(0, 0, 255),
  cv::Scalar(255, 255, 50),
  cv::Scalar(245, 150, 100),
};

static std::map<uint16_t, std::string> kitti_sem_cls_info = {
  {0, "unlabeled"},
  {1, "outlier"},
  {10, "car"},
  {11, "bicycle"},
  {13, "bus"},
  {15, "motorcycle"},
  {16, "on-rails"},
  {18, "truck"},
  {20, "other-vehicle"},
  {30, "person"},
  {31, "bicyclist"},
  {32, "motorcyclist"},
  {40, "road"},
  {44, "parking"},
  {48, "sidewalk"},
  {49, "other-ground"},
  {50, "building"},
  {51, "fence"},
  {52, "other-structure"},
  {60, "lane-marking"},
  {70, "vegetation"},
  {71, "trunk"},
  {72, "terrain"},
  {80, "pole"},
  {81, "traffic-sign"},
  {99, "other-object"},
  {252, "moving-car"},
  {253, "moving-bicyclist"},
  {254, "moving-person"},
  {255, "moving-motorcyclist"},
  {256, "moving-on-rails"},
  {257, "moving-bus"},
  {258, "moving-truck"},
  {259, "moving-other-vehicle"}
};

static std::map<uint16_t, cv::Scalar> kitti_sem_color_info = {
  {0, cv::Scalar(0, 0, 0)},
  {1, cv::Scalar(0, 0, 255)},
  {10, cv::Scalar(245, 150, 100)},
  {11, cv::Scalar(245, 230, 100)},
  {13, cv::Scalar(250, 80, 100)},
  {15, cv::Scalar(150, 60, 30)},
  {16, cv::Scalar(255, 0, 0)},
  {18, cv::Scalar(180, 30, 80)},
  {20, cv::Scalar(255, 0, 0)},
  {30, cv::Scalar(30, 30, 255)},
  {31, cv::Scalar(200, 40, 255)},
  {32, cv::Scalar(90, 30, 150)},
  {40, cv::Scalar(255, 0, 255)},
  {44, cv::Scalar(255, 150, 255)},
  {48, cv::Scalar(75, 0, 75)},
  {49, cv::Scalar(75, 0, 175)},
  {50, cv::Scalar(0, 200, 255)},
  {51, cv::Scalar(50, 120, 255)},
  {52, cv::Scalar(0, 150, 255)},
  {60, cv::Scalar(170, 255, 150)},
  {70, cv::Scalar(0, 175, 0)},
  {71, cv::Scalar(0, 60, 135)},
  {72, cv::Scalar(80, 240, 150)},
  {80, cv::Scalar(150, 240, 255)},
  {81, cv::Scalar(0, 0, 255)},
  {99, cv::Scalar(255, 255, 50)},
  {252, cv::Scalar(245, 150, 100)},
  {253, cv::Scalar(200, 40, 255)},
  {254, cv::Scalar(30, 30, 255)},
  {255, cv::Scalar(90, 30, 150)},
  {256, cv::Scalar(255, 0, 0)},
  {257, cv::Scalar(250, 80, 100)},
  {258, cv::Scalar(180, 30, 80)},
  {259, cv::Scalar(255, 0, 0)}
};

static std::string GetSemanticName(const uint16_t sem_id) {
  std::string cls_name;
  switch (sem_id)
  {
  case POLE_ID:
    cls_name = "POLE";
    break;
  
  case TRAFFIC_SIGN_ID:
    cls_name = "TRAFFIC_SIGN";
    break;
  
  case ROAD_ID:
    cls_name = "ROAD";
    break;

  case SIDEWALK_ID:
    cls_name = "SIDEWALK";
    break;
  
  case BUILDING_ID:
    cls_name = "BUILDING";
    break;
  
  case FENCE_ID:
    cls_name = "FENCE";
    break;

  default:
    cls_name = "NOT USED";
    break;
  }
  return cls_name;
}

static uint16_t GetSemanticId(const std::string& name) {
  if(name == "pole") {
    return POLE_ID;
  }
  else if(name == "traffic sign") {
    return TRAFFIC_SIGN_ID;
  }
  else if(name == "road") {
    return ROAD_ID;
  }
  else if(name == "sidewalk") {
    return SIDEWALK_ID;
  }
  else if(name == "building") {
    return BUILDING_ID;
  }
  else if(name == "fence") {
    return FENCE_ID;
  }
  else {
    return NONE_FEATURE_ID;
  }
}

static double GetSemanticResolution(const uint16_t sem_id) {
  switch (sem_id) {
    case POLE_ID:
      return POLE_RESOLUTION;
    
    case TRAFFIC_SIGN_ID:
      return TRAFFIC_SIGN_RESOLUTION;
    
    case ROAD_ID:
      return ROAD_RESOLUTION;

    case SIDEWALK_ID:
      return SIDEWALK_RESOLUTION;
    
    case BUILDING_ID:
      return BUILDING_RESOLUTION;

    case FENCE_ID:
      return FENCE_RESOLUTION;

    default:
      return 0;
  }
}

} // namespace SLIM

#endif