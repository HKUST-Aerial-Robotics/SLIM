#include "viewer.h"
#include <pangolin/gl/glvbo.h>

namespace SLIM
{

  void Viewer::Run()
  {
    const int WIDTH = 640;
    const int HEIGHT = 480;
    const int UI_WIDTH = 300;
    pangolin::CreateWindowAndBind("MainWindow", WIDTH * 2 + UI_WIDTH, HEIGHT * 2);

    // 3D Mouse handler requires depth testing to be enabled
    glEnable(GL_DEPTH_TEST);

    pangolin::CreatePanel("options").SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(UI_WIDTH));
    pangolin::Var<bool> show_trajectory("options.ShowTrajectory", true, true);
    pangolin::Var<bool> light_mode("options.LightMode", true, true);
    pangolin::Var<bool> follow_camera("options.FollowCamera", true, true);
    pangolin::Var<bool> render_sem_class("options.RenderSemClass", true, true);
    pangolin::Var<bool> show_ref_map("options.ShowRefMap", true, true);
    pangolin::Var<bool> show_cur_map("options.ShowCurMap", true, true);
    pangolin::Var<bool> show_buff_map("options.ShowBufferMap", true, true);
    pangolin::Var<bool> show_connection("options.ShowConnection", false, false);
    pangolin::Var<bool> show_obs("options.ShowObs", false, false);    
    pangolin::Var<bool> show_triplet("options.ShowTriplet", true, true);  
    pangolin::Var<bool> show_loop("options.ShowLoop", true, true);
    pangolin::Var<bool> show_host_frame("options.ShowHostFrame", true, true);
    pangolin::Var<bool> show_cloud("options.ShowCloud", true, true);
    pangolin::Var<bool> show_semantic_graph("options.RenderSemanticGraph", true, true);

    pangolin::Var<float> render_point_size("options.RenderPointSize", 2.0, 0.1, 10.0);
    pangolin::Var<float> render_point_alpha("options.RenderPointAlpha", 0.4, 0.0, 1.0);
    pangolin::Var<float> render_line_width("options.RenderLineWidth", 4.0, 0.1, 10.0);
    // Define Camera Render Object (for view / scene browsing)
    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(WIDTH, HEIGHT, 400, 400, WIDTH / 2, HEIGHT / 2, 0.1, 1e6),
        pangolin::ModelViewLookAt(-10, 0, 0, 0, 0, 0, pangolin::AxisZ));

    // Add named OpenGL viewport to window and provide 3D Handler
    pangolin::View &d_cam = pangolin::Display("cam")
                                .SetBounds(0.0, 1.0, pangolin::Attach::Pix(UI_WIDTH), 1.0, -WIDTH / (float)HEIGHT)
                                .SetHandler(new pangolin::Handler3D(s_cam));

    pangolin::View &d_color = pangolin::Display("Color")
                                  .SetAspect(WIDTH / (float)HEIGHT);

    pangolin::CreateDisplay()
        .SetBounds(0.0, 0.2, pangolin::Attach::Pix(UI_WIDTH), 1.0)
        .SetLayout(pangolin::LayoutEqual)
        .AddDisplay(d_color);

    bool curFollowFlag = false;

    while (!pangolin::ShouldQuit())
    {
      if(stop_)
        break;

      auto Twc = GetPGLCameraPose();
      if(follow_camera && curFollowFlag) {
        s_cam.Follow(Twc);
      }
      else if(follow_camera && !curFollowFlag) {
        s_cam.SetModelViewMatrix(pangolin::ModelViewLookAt(0,-0.7,-1.8, 0,0,0,0.0,-1.0, 0.0));
        s_cam.Follow(Twc);
        curFollowFlag = true;
      }
      else if(!follow_camera && curFollowFlag) {
        curFollowFlag = false;
      }
      
      // Clear screen and activate view to render into
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
      if(light_mode)
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
      else 
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
      d_cam.Activate(s_cam);

      pangolin::glDrawAxis(1.0);
      DrawAxis(Transform(twc_, Rwc_), 2.0);
      // Render Pipeline
      if(show_ref_map)
        DrawVectorMap(ref_map_, render_line_width, render_sem_class, show_connection, show_obs, 0);
      if(show_cur_map)
        DrawVectorMap(cur_map_, render_line_width, render_sem_class, show_connection, show_obs, 1);
      
      if(show_buff_map) {
        for(uint32_t id = 0; id < vec_map_.size(); ++id) {
          DrawVectorMap(vec_map_[id], render_line_width, render_sem_class, show_connection, show_obs, id);
        }        
      }

      if(show_loop)
        DrawLoop();
      // if(show_host_frame)
      //   DrawLoopCandidate();

      // graph matching
      // DrawCloud(render_point_size, render_point_alpha);
      // if(show_semantic_graph)
      //   DrawMapMatching();
      // Swap frames and Process Events
      pangolin::FinishFrame();
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    printf("Distroy Window\n");
  }

  pangolin::OpenGlMatrix Viewer::GetPGLCameraPose()
  {
    pangolin::OpenGlMatrix T;
    T.SetIdentity();
    if (pose_ok_)
    {
      T.m[0] = Rwc_(0, 0);
      T.m[1] = Rwc_(1, 0);
      T.m[2] = Rwc_(2, 0);
      T.m[3] = 0.0;
      T.m[4] = Rwc_(0, 1);
      T.m[5] = Rwc_(1, 1);
      T.m[6] = Rwc_(2, 1);
      T.m[7] = 0.0;
      T.m[8] = Rwc_(0, 2);
      T.m[9] = Rwc_(1, 2);
      T.m[10] = Rwc_(2, 2);
      T.m[11] = 0.0;
      T.m[12] = twc_(0);
      T.m[13] = twc_(1);
      T.m[14] = twc_(2);
      T.m[15] = 1.0;
    }
    return T;
  }

  void Viewer::AddCamera(const Eigen::Matrix4d &Twc, float r, float g, float b, float lw, float scale)
  {
    glPushMatrix();

    pangolin::OpenGlMatrix T;
    T.SetIdentity();

    T.m[0] = Twc(0, 0);
    T.m[1] = Twc(1, 0);
    T.m[2] = Twc(2, 0);
    T.m[3] = 0.0;
    T.m[4] = Twc(0, 1);
    T.m[5] = Twc(1, 1);
    T.m[6] = Twc(2, 1);
    T.m[7] = 0.0;
    T.m[8] = Twc(0, 2);
    T.m[9] = Twc(1, 2);
    T.m[10] = Twc(2, 2);
    T.m[11] = 0.0;
    T.m[12] = Twc(0, 3);
    T.m[13] = Twc(1, 3);
    T.m[14] = Twc(2, 3);
    T.m[15] = 1.0;

    glMultMatrixd(T.m);
    const float w = 0.2 * scale;
    const float h = w * 0.75 * scale;
    const float z = w * scale;

    glLineWidth(lw);
    glBegin(GL_LINES);
    glColor3f(r, g, b);
    glVertex3f(0, 0, 0);
    glVertex3f(w, h, z);
    glVertex3f(0, 0, 0);
    glVertex3f(w, -h, z);
    glVertex3f(0, 0, 0);
    glVertex3f(-w, -h, z);
    glVertex3f(0, 0, 0);
    glVertex3f(-w, h, z);
    glVertex3f(w, h, z);
    glVertex3f(w, -h, z);
    glVertex3f(-w, h, z);
    glVertex3f(-w, -h, z);
    glVertex3f(-w, h, z);
    glVertex3f(w, h, z);
    glVertex3f(-w, -h, z);
    glVertex3f(w, -h, z);
    glEnd();
    glPopMatrix();
  }

  void Viewer::DrawCamera(const Eigen::Matrix3d &Rwc, const Eigen::Vector3d &twc)
  {
    Eigen::Matrix4d Twc = Eigen::Matrix4d::Identity();
    Twc.topLeftCorner(3, 3) = Rwc;
    Twc.topRightCorner(3, 1) = twc;

    AddCamera(Twc, 0, 1, 1, 2);

    trajectory.push_back(twc);

    if (trajectory.size() <= 1)
      return;

    glLineWidth(1);
    glBegin(GL_LINES);
    for (size_t i = 0; i < trajectory.size() - 1; i++)
    {
      glColor3f(0.0f, 1.0f, 0.0f);
      glVertex3f(trajectory[i].x(), trajectory[i].y(), trajectory[i].z());
      glVertex3f(trajectory[i + 1].x(), trajectory[i + 1].y(), trajectory[i + 1].z());
    }
    glEnd();
  }

  void Viewer::DrawAxis(const Transform &transform, const double scale)
  {
    Eigen::Matrix3d const Rwb = transform.dcm();
    Eigen::Vector3d const twb = transform.p();

    std::vector<Eigen::Vector3d> uvec(3);
    uvec[0] = (twb + scale * Rwb.col(0));
    uvec[1] = (twb + scale * Rwb.col(1));
    uvec[2] = (twb + scale * Rwb.col(2));

    glLineWidth(5);
    glBegin(GL_LINES);

    glColor3f(1.0f, 0.0f, 0.0f);
    glVertex3f(twb.x(), twb.y(), twb.z());
    glVertex3f(uvec[0].x(), uvec[0].y(), uvec[0].z());

    glColor3f(0.0f, 1.0f, 0.0f);
    glVertex3f(twb.x(), twb.y(), twb.z());
    glVertex3f(uvec[1].x(), uvec[1].y(), uvec[1].z());

    glColor3f(0.0f, 0.0f, 1.0f);
    glVertex3f(twb.x(), twb.y(), twb.z());
    glVertex3f(uvec[2].x(), uvec[2].y(), uvec[2].z());

    glEnd();
  }

  void Viewer::DrawVectorMap(const VectorMap::Ptr map, const float lw, const bool color_flag, const bool show_connection, const bool show_obs, const uint32_t id) {

    if(map.get() == nullptr)
      return;

    // std::lock_guard<std::mutex> lock(graph_matching_mutex_);

    std::map<uint32_t, Transform> poses, gt_poses;
    std::vector<std::pair<uint32_t, uint32_t>> rel_pose_infos;
    std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> poles;
    std::vector<std::vector<Eigen::Vector3d>> roads, buildings;
    std::vector<std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>>> pole_obs;
    std::vector<std::vector<std::vector<Eigen::Vector3d>>> road_obs, building_obs;

    map->visualize(poses, gt_poses, rel_pose_infos, poles, roads, buildings, pole_obs, road_obs, building_obs);
    for(auto pose: poses) {
      DrawAxis(pose.second, 2.0);
    }
    for(auto pose: gt_poses) {
      DrawAxis(pose.second, 1.0);
    }
    for(auto &info: rel_pose_infos) {
      Eigen::Vector3d node_i = poses[info.first].p();
      Eigen::Vector3d node_j = poses[info.second].p();
      cv::Scalar color = cs_map_color[id%cs_map_color.size()];
      DrawLine(node_i, node_j, lw * 2, CV_COLOR_INDIGO);
    }

    for(auto &line: poles) {
      glLineWidth(10);
      glBegin(GL_LINES);
      glColor3f((float)CV_COLOR_SKYBLUE[2]/255, (float)CV_COLOR_SKYBLUE[1]/255, (float)CV_COLOR_SKYBLUE[0]/255);
      auto const& pt = line.first;
      auto const& pb = line.second;
      glVertex3f(pt.x(), pt.y(), pt.z());
      glVertex3f(pb.x(), pb.y(), pb.z());
      glEnd();
    }

    for(auto &surf: roads) {
      cv::Scalar color = CV_COLOR_DEEPPINK;
      glColor3f((float)255/255, (float)0/255, (float)255/255);
      pangolin::GlBuffer vbo1(pangolin::GlArrayBuffer,
            std::vector<Eigen::Vector3f>{
              surf[0].cast<float>(),
              surf[1].cast<float>(),
              surf[3].cast<float>(),
            }
        );
      pangolin::RenderVbo(vbo1, GL_TRIANGLES);
      pangolin::GlBuffer vbo2(pangolin::GlArrayBuffer,
            std::vector<Eigen::Vector3f>{
              surf[0].cast<float>(),
              surf[1].cast<float>(),
              surf[2].cast<float>(),
            }
        );
      pangolin::RenderVbo(vbo2, GL_TRIANGLES);
    }

    for(auto &surf: buildings) {
      cv::Scalar color = CV_COLOR_ORANGERED;
      glColor3f((float)color[2]/255, (float)color[1]/255, (float)color[0]/255);
      pangolin::GlBuffer vbo1(pangolin::GlArrayBuffer,
            std::vector<Eigen::Vector3f>{
              surf[0].cast<float>(),
              surf[1].cast<float>(),
              surf[3].cast<float>(),
            }
        );
      pangolin::RenderVbo(vbo1, GL_TRIANGLES);
      pangolin::GlBuffer vbo2(pangolin::GlArrayBuffer,
            std::vector<Eigen::Vector3f>{
              surf[0].cast<float>(),
              surf[1].cast<float>(),
              surf[2].cast<float>(),
            }
        );
      pangolin::RenderVbo(vbo2, GL_TRIANGLES);
    }
  }

  void Viewer::DrawTrajectory(const std::map<uint64_t, Frame::Ptr>& traj, const uint32_t id) 
  {
    std::vector<Frame::Ptr> path;
    for(auto traj_iter = traj.begin(); traj_iter != traj.end(); ++traj_iter) {
      path.push_back(traj_iter->second);
    }
    for(int i = 0; i < path.size() - 1; ++i) {
      auto Pwbi = path[i]->Twb().p();
      auto Pwbj = path[i+1]->Twb().p();
      cv::Scalar color = cs_map_color[id%cs_map_color.size()];

      glLineWidth(5);
      glBegin(GL_LINES);
      glColor3f((float)color[2]/255, (float)color[1]/255, (float)color[0]/255);
      glVertex3f(Pwbi.x(), Pwbi.y(), Pwbi.z());
      glVertex3f(Pwbj.x(), Pwbj.y(), Pwbj.z());
      glEnd();
    }
  }

  void Viewer::DrawCurFrame(const float width)
  {
    std::unique_lock<std::mutex> lock(frame_mutex_);

    if(frame_buffer_.empty())
      return;

    auto const frame = frame_buffer_.back();
    auto const Twb = frame->Twb();

    auto const &line_obs = frame->line_obs();
    auto const &surface_obs = frame->surface_obs();

    DrawAxis(Twb);

    cv::Scalar color;
    color = kitti_sem_color_info[POLE_ID];
    Eigen::Vector3f pcolor((float)CV_COLOR_SKYBLUE[2] / 255, (float)CV_COLOR_SKYBLUE[1] / 255, (float)CV_COLOR_SKYBLUE[0] / 255);

    for (int j = 0; j < line_obs.size(); ++j)
    {
      Eigen::Vector3d const point_a = Twb * line_obs[j]->point_a();
      Eigen::Vector3d const point_b = Twb * line_obs[j]->point_b();

      glLineWidth(width);
      glBegin(GL_LINES);
      glColor3f(pcolor.x(), pcolor.y(), pcolor.z());
      glVertex3f(point_a.x(), point_a.y(), point_a.z());
      glVertex3f(point_b.x(), point_b.y(), point_b.z());
      glEnd();
    }

    for (int j = 0; j < surface_obs.size(); ++j)
    {
      std::vector<Eigen::Vector3d> vertices = surface_obs[j]->vertices();
      for (auto &vertex : vertices)
      {
        vertex.noalias() = Twb * vertex;
      }

      glLineWidth(width);
      glBegin(GL_LINES);

      color = kitti_sem_color_info[surface_obs[j]->semantic_type()];
      Eigen::Vector3f pcolor((float)color[2] / 255, (float)color[1] / 255, (float)color[0] / 255);
      glColor3f(pcolor.x(), pcolor.y(), pcolor.z());

      glVertex3f(vertices[0].x(), vertices[0].y(), vertices[0].z());
      glVertex3f(vertices[1].x(), vertices[1].y(), vertices[1].z());

      glVertex3f(vertices[1].x(), vertices[1].y(), vertices[1].z());
      glVertex3f(vertices[2].x(), vertices[2].y(), vertices[2].z());

      // glVertex3f(vertices[2].x(), vertices[2].y(), vertices[2].z());
      // glVertex3f(vertices[3].x(), vertices[3].y(), vertices[3].z());

      // glVertex3f(vertices[3].x(), vertices[3].y(), vertices[3].z());
      // glVertex3f(vertices[0].x(), vertices[0].y(), vertices[0].z());

      glVertex3f(vertices[2].x(), vertices[2].y(), vertices[2].z());
      glVertex3f(vertices[0].x(), vertices[0].y(), vertices[0].z());

      glEnd();
    }
  }

  void Viewer::DrawGraphMatching() {
    std::unique_lock<std::mutex> lock(graph_matching_mutex_);
    if(graph_a_.get() == nullptr || graph_b_.get() == nullptr)
      return;

    for(int i = 0; i < graph_a_->nodes_.size(); ++i) {
      SceneGraphNode::Ptr node = graph_a_->nodes_[i];
      cv::Scalar color = random_color_vec[i];
      if(node->semantic_type() == POLE_ID) {
        LineOB::Ptr ft = node->vec_line_feature_ptr_.front();
        Eigen::Vector3d const point_a = ft->point_a();
        Eigen::Vector3d const point_b = ft->point_b();
        glLineWidth(3);
        glBegin(GL_LINES);
        glColor3f((float)color[2]/255, (float)color[1]/255, (float)color[0]/255);
        glVertex3f(point_a(0), point_a(1), point_a(2));
        glVertex3f(point_b(0), point_b(1), point_b(2));
        glEnd();        
      }
      else {
        Eigen::Vector3d const centroid = node->node_value();
        glPointSize(10);
        glBegin(GL_POINTS);
        glColor3f((float)color[2]/255, (float)color[1]/255, (float)color[0]/255);
        glVertex3f(centroid(0), centroid(1), centroid(2));
        glEnd();  
      }
    }

    for(int i = 0; i < graph_b_->nodes_.size(); ++i) {
      SceneGraphNode::Ptr node = graph_b_->nodes_[i];
      cv::Scalar color = random_color_vec[i];
      if(node->semantic_type() == POLE_ID) {
        LineOB::Ptr ft = node->vec_line_feature_ptr_.front();
        Eigen::Vector3d const point_a = ft->point_a();
        Eigen::Vector3d const point_b = ft->point_b();
        glLineWidth(3);
        glBegin(GL_LINES);
        glColor3f((float)color[2]/255, (float)color[1]/255, (float)color[0]/255);
        glVertex3f(point_a(0), point_a(1), point_a(2) + 10);
        glVertex3f(point_b(0), point_b(1), point_b(2) + 10);
        glEnd();        
      }
      else {
        Eigen::Vector3d const centroid = node->node_value();
        glPointSize(10);
        glBegin(GL_POINTS);
        glColor3f((float)color[2]/255, (float)color[1]/255, (float)color[0]/255);
        glVertex3f(centroid(0), centroid(1), centroid(2) + 10);
        glEnd();  

        // for(auto const ft: node->vec_surf_feature_ptr_) {
        //   Eigen::Vector3d const ftc = ft->centroid();
        //   glPointSize(6);
        //   glBegin(GL_POINTS);
        //   glColor3f((float)color[2]/255, (float)color[1]/255, (float)color[0]/255);
        //   glVertex3f(ftc(0), ftc(1), ftc(2) + 10);
        //   glEnd();  
        // }
      }
    }

    for(auto pair: graph_match_) {
      uint32_t id_a = pair.first;
      uint32_t id_b = pair.second;
      SceneGraphNode::Ptr node_a = graph_a_->nodes_[id_a];
      SceneGraphNode::Ptr node_b = graph_b_->nodes_[id_b];
      Eigen::Vector3d const point_a = node_a->node_value();
      Eigen::Vector3d const point_b = node_b->node_value();
      glLineWidth(1);
      glBegin(GL_LINES);
      glColor3f(0, 0, 0);
      glVertex3f(point_a(0), point_a(1), point_a(2));
      glVertex3f(point_b(0), point_b(1), point_b(2) + 10);
      glEnd();   
    }
  }

  void Viewer::DrawMapMatching() {
    std::unique_lock<std::mutex> lock(graph_matching_mutex_);
    if(graph_a_.get() != nullptr)
      DrawSemGraph(graph_a_, 0);
    if(graph_b_.get() != nullptr)
      DrawSemGraph(graph_b_, 1);
    if(!graph_match_.empty())
      DrawSemMatch(graph_a_, graph_b_, graph_match_);
  }

  void Viewer::DrawLoopCandidate() {
    std::unique_lock<std::mutex> lock(graph_matching_mutex_);
    for(auto node: ref_blocks_) {
      Eigen::Vector3d c = node->getHostFrame()->Twb().p();
      glPointSize(10);
      glBegin(GL_POINTS);
      glColor3f((float)CV_COLOR_CHOCOLATE[2]/255, (float)CV_COLOR_CHOCOLATE[1]/255, (float)CV_COLOR_CHOCOLATE[0]/255);
      glVertex3f(c(0), c(1), c(2));
      glEnd();
    }
    for(auto node: cur_blocks_) {
      Eigen::Vector3d c = node->getHostFrame()->Twb().p();
      glPointSize(10);
      glBegin(GL_POINTS);
      glColor3f((float)CV_COLOR_CHOCOLATE[2]/255, (float)CV_COLOR_CHOCOLATE[1]/255, (float)CV_COLOR_CHOCOLATE[0]/255);
      glVertex3f(c(0), c(1), c(2));
      glEnd();
    }
  }

  void Viewer::DrawLoop() {
    for(auto &loop: outlier_loop_info_) {
      if(loop.first == nullptr || loop.second == nullptr)
        continue;
      
      Eigen::Vector3d const pi = loop.first->Twb().p();
      Eigen::Vector3d const pj = loop.second->Twb().p();

      glPointSize(12);
      glBegin(GL_POINTS);
      glColor3f((float)CV_COLOR_RED[2]/255, (float)CV_COLOR_RED[1]/255, (float)CV_COLOR_RED[0]/255);
      glVertex3f(pi(0), pi(1), pi(2));
      glEnd();   

      glPointSize(12);
      glBegin(GL_POINTS);
      glColor3f((float)CV_COLOR_RED[2]/255, (float)CV_COLOR_RED[1]/255, (float)CV_COLOR_RED[0]/255);
      glVertex3f(pj(0), pj(1), pj(2));
      glEnd(); 

      glLineWidth(10);
      glBegin(GL_LINES);
      glColor3f((float)CV_COLOR_RED[2]/255, (float)CV_COLOR_RED[1]/255, (float)CV_COLOR_RED[0]/255);
      glVertex3f(pi(0), pi(1), pi(2));
      glVertex3f(pj(0), pj(1), pj(2));
      glEnd();    
    }

    for(auto &loop: loop_info_) {
      if(loop.first == nullptr || loop.second == nullptr)
        continue;
      
      Eigen::Vector3d const pi = loop.first->Twb().p();
      Eigen::Vector3d const pj = loop.second->Twb().p();

      glPointSize(12);
      glBegin(GL_POINTS);
      glColor3f((float)CV_COLOR_GREEN[2]/255, (float)CV_COLOR_GREEN[1]/255, (float)CV_COLOR_GREEN[0]/255);
      glVertex3f(pi(0), pi(1), pi(2));
      glEnd();   

      glPointSize(12);
      glBegin(GL_POINTS);
      glColor3f((float)CV_COLOR_GREEN[2]/255, (float)CV_COLOR_GREEN[1]/255, (float)CV_COLOR_GREEN[0]/255);
      glVertex3f(pj(0), pj(1), pj(2));
      glEnd();   

      glLineWidth(10);
      glBegin(GL_LINES);
      glColor3f((float)CV_COLOR_GREEN[2]/255, (float)CV_COLOR_GREEN[1]/255, (float)CV_COLOR_GREEN[0]/255);
      glVertex3f(pi(0), pi(1), pi(2));
      glVertex3f(pj(0), pj(1), pj(2));
      glEnd();  
    }
  }

  void Viewer::DrawSemGraph(const SceneGraph::Ptr graph, int level, bool show_triplet) {
    for(int i = 0; i < graph->nodes_.size(); ++i) {
      SceneGraphNode::Ptr node = graph->nodes_[i];
      cv::Scalar color = random_color_vec[i%random_color_vec.size()];
      if(node->semantic_type() == POLE_ID) {
        // auto lm = node->vec_line_lms_.front();
        // auto obs = lm->getAllObs();
        Eigen::Vector3d const centroid = node->node_value();
        Eigen::Vector3d const normal = node->normal_value();
        Eigen::Vector3d const point_a = centroid + normal * 2;
        Eigen::Vector3d const point_b = centroid - normal * 2;

        glLineWidth(8);
        glBegin(GL_LINES);
        glColor3f((float)color[2]/255, (float)color[1]/255, (float)color[0]/255);
        glVertex3f(point_a(0), point_a(1) + level * 0, point_a(2) + level * 40);
        glVertex3f(point_b(0), point_b(1) + level * 0, point_b(2) + level * 40);
        glEnd();  

        // glPointSize(15);
        // glBegin(GL_POINTS);
        // glColor3f((float)color[2]/255, (float)color[1]/255, (float)color[0]/255);
        // glVertex3f(centroid(0), centroid(1), centroid(2) + level * 40);
        // glEnd();          
      }
      else {
        Eigen::Vector3d const centroid = node->node_value();
        Eigen::Vector3d const normal = node->normal_value();
        // glPointSize(15);
        // glBegin(GL_POINTS);
        // glColor3f((float)color[2]/255, (float)color[1]/255, (float)color[0]/255);
        // glVertex3f(centroid(0), centroid(1), centroid(2) + level * 40);
        // glEnd();  
        int node_size = node->vec_nodes_.size();
        if(node_size < 3)
          continue;

        for(auto const m: node->vec_nodes_) {
          // Eigen::Vector3d const ftc = lm->centroid();
          std::vector<Eigen::Vector3d> vertices;
          for(auto& vertex: m->vertices) {
            vertices.push_back(Eigen::Vector3d(vertex.x(), vertex.y() + level * 0, vertex.z() + level * 40));
          }
        
          glColor3f((float)color[2]/255, (float)color[1]/255, (float)color[0]/255);
          pangolin::GlBuffer vbo1(pangolin::GlArrayBuffer,
                std::vector<Eigen::Vector3f>{
                  vertices[0].cast<float>(),
                  vertices[1].cast<float>(),
                  vertices[3].cast<float>(),
                }
            );
          pangolin::RenderVbo(vbo1, GL_TRIANGLES);
          pangolin::GlBuffer vbo2(pangolin::GlArrayBuffer,
                std::vector<Eigen::Vector3f>{
                  vertices[0].cast<float>(),
                  vertices[1].cast<float>(),
                  vertices[2].cast<float>(),
                }
            );
          pangolin::RenderVbo(vbo2, GL_TRIANGLES);
          
          // glPointSize(6);
          // glBegin(GL_POINTS);
          // glColor3f((float)color[2]/255, (float)color[1]/255, (float)color[0]/255);
          // glVertex3f(ftc(0), ftc(1) + level * 40, ftc(2) + level * 40);
          // glEnd();  
        }

        // for(auto const node: node->vec_nodes_) {
        //   Eigen::Vector3d const& ftc = node->centroid;
        //   glPointSize(6);
        //   glBegin(GL_POINTS);
        //   glColor3f((float)color[2]/255, (float)color[1]/255, (float)color[0]/255);
        //   glVertex3f(ftc(0), ftc(1) + level * 40, ftc(2) + level * 40);
        //   glEnd();  
        // }
      }
    }
  }

  void Viewer::DrawSemMatch(const SceneGraph::Ptr graph_a, const SceneGraph::Ptr graph_b, const std::vector<std::pair<uint32_t, uint32_t>>& matches) {
    for(auto pair: matches) {
      uint32_t id_a = pair.first;
      uint32_t id_b = pair.second;
      SceneGraphNode::Ptr node_a = graph_a->nodes_[id_a];
      SceneGraphNode::Ptr node_b = graph_b->nodes_[id_b];
      Eigen::Vector3d const point_a = node_a->node_value();
      Eigen::Vector3d const point_b = node_b->node_value();
      glLineWidth(1);
      glBegin(GL_LINES);
      glColor3f(0, 0, 0);
      glVertex3f(point_a(0), point_a(1), point_a(2));
      glVertex3f(point_b(0), point_b(1) + 0, point_b(2) + 40);
      glEnd();   
    }
  }

  void Viewer::DrawLine(const Eigen::Vector3d& p1, const Eigen::Vector3d& p2,
                        const double lw, const cv::Scalar color) {
    Eigen::Vector3f pcolor((float)color[2]/255, (float)color[1]/255, (float)color[0]/255);
    glLineWidth(lw);
    glBegin(GL_LINES);
    glVertex3f(p1.x(), p1.y(), p1.z());
    glVertex3f(p2.x(), p2.y(), p2.z());
    glEnd();
  }

  void Viewer::DrawTriangle(const Eigen::Vector3d& p1, const Eigen::Vector3d& p2,
                  const Eigen::Vector3d& p3, const double lw, const cv::Scalar color) {
    Eigen::Vector3f pcolor((float)color[2]/255, (float)color[1]/255, (float)color[0]/255);
    glLineWidth(lw);
    glBegin(GL_LINES);
    glColor3f(pcolor.x(), pcolor.y(), pcolor.z());
    glVertex3f(p1.x(), p1.y(), p1.z());
    glVertex3f(p2.x(), p2.y(), p2.z());
    glVertex3f(p2.x(), p2.y(), p2.z());
    glVertex3f(p3.x(), p3.y(), p3.z());
    glVertex3f(p3.x(), p3.y(), p3.z());
    glVertex3f(p1.x(), p1.y(), p1.z());
    glEnd();
  }

  void Viewer::DrawDiamond(const Eigen::Vector3d& p1, const Eigen::Vector3d& p2,
                           const Eigen::Vector3d& p3, const Eigen::Vector3d& p4, 
                           const double lw, const cv::Scalar color) {
    Eigen::Vector3f pcolor((float)color[2]/255, (float)color[1]/255, (float)color[0]/255);
    glLineWidth(lw);
    glBegin(GL_LINES);
    glColor3f(pcolor.x(), pcolor.y(), pcolor.z());
    glVertex3f(p1.x(), p1.y(), p1.z());
    glVertex3f(p2.x(), p2.y(), p2.z());
    glVertex3f(p2.x(), p2.y(), p2.z());
    glVertex3f(p3.x(), p3.y(), p3.z());
    glVertex3f(p3.x(), p3.y(), p3.z());
    glVertex3f(p4.x(), p4.y(), p4.z());
    glVertex3f(p4.x(), p4.y(), p4.z());
    glVertex3f(p1.x(), p1.y(), p1.z());
    glEnd();
  }
} // namespace SLIM
