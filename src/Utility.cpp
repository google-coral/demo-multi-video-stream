/*
 * Copyright 2021 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
/*
 * Polygons.cpp
 *
 *  Created on: Sep 9, 2021
 *      Author: pnordstrom
 */

#include <cmath>
#include <string>
#include <utility>
#include <vector>

#include "Utility.h"

namespace szd {

const double Utility::Point::GetDistance(const Point &p) const {
  return sqrt(pow((x_ - p.x_), 2) + pow((y_ - p.y_), 2));
}
const int Utility::Point::GetDirection(const Point &b, const Point &c) const {
  const float cross_product = (b.y_ - y_) * (c.x_ - b.x_)
      - (b.x_ - x_) * (c.y_ - b.y_);
  if (std::abs(cross_product) < EPSILON) {
    return 0;
  }
  return (cross_product > 0) ? 1 : 2;
}

bool Utility::Line::ContainsPoint(const Point &p) const {
  // The distance between begin_ to p + p to end_ should equal to length of
  // this line: begin_----------p-----end_
  return ((begin_.GetDistance(p) + end_.GetDistance(p)) - length_) < EPSILON;
}
bool Utility::Line::IntersectsLine(const Line &l) const {
  int dir1 = begin_.GetDirection(end_, l.begin_);
  int dir2 = begin_.GetDirection(end_, l.end_);
  int dir3 = l.begin_.GetDirection(l.end_, begin_);
  int dir4 = l.begin_.GetDirection(l.end_, end_);
  if (dir1 != dir2 && dir3 != dir4)
    return true;  // These 2 lines are intersecting.
  if (dir1 == 0 && ContainsPoint(l.begin_))
    return true;
  if (dir2 == 0 && ContainsPoint(l.end_))
    return true;
  if (dir3 == 0 && l.ContainsPoint(begin_))
    return true;
  if (dir4 == 0 && l.ContainsPoint(end_))
    return true;
  return false;
}

Utility::Polygon::Polygon(std::vector<Point> &polygon_points) {
  lines_.emplace_back(polygon_points[0],
                      polygon_points[polygon_points.size() - 1]);
  for (size_t i = 0; i < polygon_points.size() - 1; i++)
    lines_.emplace_back(polygon_points[i], polygon_points[i + 1]);
}

Utility::Box::Box(float x1, float y1, float x2, float y2) {
  points_.emplace_back(x1, y1);
  points_.emplace_back(x1, y2);
  points_.emplace_back(x2, y1);
  points_.emplace_back(x2, y2);
  lines_.emplace_back(points_[0], points_[1]);
  lines_.emplace_back(points_[1], points_[2]);
  lines_.emplace_back(points_[2], points_[3]);
  lines_.emplace_back(points_[3], points_[0]);
}

const bool Utility::Box::IntersectsLine(const Line &l) const {
  for (const auto &line : lines_) {
    if (line.IntersectsLine(l)) {
      return true;
    }
  }
  return false;
}

const bool Utility::Box::CollidedWithPolygon(const Polygon &p,
                                             const float max_width) const {
  const auto &polygon_lines = p.GetLines();
  for (const auto &p : points_) {
    size_t intersect_time { 0 };
    // We create a horizontal line from this point to max image width, if
    // it interects the lines in the polygon even time, it is not inside
    // the polygon. If it is odd, it is inside the polygon.
    Line extreme { p, { max_width, p.y_ } };
    for (const auto &l : polygon_lines) {
      // If p is on any of the lines, it is a collision.
      if (l.ContainsPoint(p)) {
        return true;
      }
      // If this line in the polygon intersects the extreme line.
      if (l.IntersectsLine(extreme)) {
        intersect_time++;
      }
    }
    if (intersect_time % 2 == 1) {
      return true;
    }
  }
  // Check for line intersection between this box and the polygon.
  for (const auto &box_line : lines_) {
    for (const auto &polygon_line : p.GetLines()) {
      if (box_line.IntersectsLine(polygon_line)) {
        return true;
      }
    }
  }
  return false;
}

Utility::Utility() {
}

Utility::~Utility() {
}

} /* namespace szd */
