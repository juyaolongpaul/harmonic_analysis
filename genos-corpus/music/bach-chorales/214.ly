\version "2.10.33"

\header {
  title = "214 - Mitten wir im Leben sind"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key a \minor
}


soprano = \relative c'' {
  \repeat volta 2 {
    g4 g4 a4 b4 
    c4 c4 b2 
    a2 b4 c4 
    d4 a4 g4 f4 
    e1 }  %5
  r4 d4 e4 f4 
  g4. f8 e4 d4 
  c2 r4 c4 
  g'4 g4 a4 b4 
  c4 c4 b2   %10
  a2 b4 c4 
  d4 a4 g4 f4 
  e2 r4 a4 
  g4 e4 f4 g8. f16 
  e2 r4 a4   %15
  g4 e4 f4 g4 
  e2 a4 g4 
  a4 b4 c4 b8 a8 
  g4 f4 e4 d4 
  a'2 g4 f4   %20
  e2 a4 g4 
  a4 b4 c8 b8 a4 
  g2 c4 b4 
  d4 a4 g4 f4 
  e2 r4 d4   %25
  a'2 g4 f4 
  e1 
  
}

alto = \relative c' {
  \repeat volta 2 {
    e4 e4 e4 e4 
    e8 fis16 gis16 a4 a4 gis4 
    e2 g4 g4 
    a8 g8 f4. e4 d16 c16 
    b1 }  %5
  r4 b4 c8 b8 c4 
  d8 e4 d4 c4 b8 
  g2 r4 g4 
  c4. e8 f4. e8 
  e8 a16 gis16 a4. gis16 fis16 gis4   %10
  e2 gis4 a4~ 
  a8 g8 f4 e4. d8 
  cis2 r4 f8 e8 
  d4. c8 d4 d4 
  c2 r4 e4   %15
  d4. c16 b16 a4 d8 b8 
  c2 e4 e4 
  e4 e4 e8 f8 g8 f8~ 
  f8 e4 d16 c16 b4 bes4 
  c4 f4. e4 d8   %20
  c2 e4. d8 
  c4 d4 e8 g4 fis8 
  g2 g4 g4 
  a8 g8 f8 e8 d8 e8 a,8 d8 
  cis2 r4 d4   %25
  d4 c8 d8 e4. d16 c16 
  b1 
  
}

tenor = \relative c' {
  \repeat volta 2 {
    b4 b4 c4 b4 
    a4. e'8 f8 d8 b8 e16 d16 
    c2 d4 e4 
    f8 e8 d4. c16 bes16 a8 b16 a16 
    gis1 }    %5
    r4 g4 g4 c8 a8 
  b4 c8 g8 g4~ g8. f16 
  e2 r4 g4 
  g4 c4 c8 d8 d4 
  c8 d8 e4. b8 e8 d8   %10
  c2 e4 e4 
  d4. c8 bes4 a4 
  a2 r4 c4~ 
  c8 b8 a4 a4 g8 b8 
  c2 r4 c4   %15
  b4 g4 f8 e8 d8 g8 
  g2 c4 b4 
  c4 b4 a4 d4 
  g,8 c16 b16 a8 b16 a16 gis4 g4 
  a4 d4. b8 c8 g8   %20
  g2 c4 c8 bes8 
  a8 g8 f4 g8 e'8 a,8 d16 c16 
  b2 e4 d8 e8 
  f8 e8 d8 c8 b8 cis8 d8 a8 
  a2 r4 a8 g8   %25
  f8 e8 f4 e8 a8 a8 b16 a16 
  gis1 
  
}

baixo = \relative c {
  \repeat volta 2 {
    e4 e8 d8 c8 b8 a8 gis8 
    a4. c8 d8 b8 e4 
    a,2 g'8 f8 e4 
    d8 e8 f4 b,8 c8 d4 
    e1 }  %5
  r4 g8 f8 e4 a4 
  g8 g,8 a8 b8 c4 g4 
  c,2 r4 e'8 f8 
  e8 d8 e8 c8 f4 gis,4 
  a8 b8 c8 d8 e2   %10
  a,2 e'4 a8 g8 
  f8 e8 d4. cis8 d8 f8 
  a2 r4 f,4 
  g4 a4 d8 c8 b8 g8 
  a2 r4 a4   %15
  b4 c4 d4 b8 g8 
  c2 a8 c8 e8 d8 
  c8 b8 a8 gis8 a4 b4 
  c4 d4 e4 g4 
  f8 e8 d8 c8 b8 g8 a8 b8   %20
  c2 c8 d8 e4 
  f8 e8 d4 c4 d4 
  e2 e8 f8 g4 
  d8 e8 f4. e8 d8 f8 
  a2 r4 f8 e8   %25
  d4 a8 b8 c8 cis8 d4 
  e1 
  
}
\score {
  <<
    \new StaffGroup <<
      \override StaffGroup.SystemStartBracket #'style = #'line 
      \new Staff {
        <<
          \global
          \new Voice = "soprano" { \voiceOne \soprano }
          \new Voice = "alto" { \voiceTwo \alto }
        >>
      }
      \new Staff {
        <<
          \global
          \clef "bass"
          \new Voice = "tenor" {\voiceOne \tenor }
          \new Voice = "baixo" { \voiceTwo \baixo \bar "|."}
        >>
      }
    >>
  >>
  \layout {}
  \midi {}
}