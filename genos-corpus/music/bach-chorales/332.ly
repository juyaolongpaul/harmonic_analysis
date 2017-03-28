\version "2.10.33"

\header {
  title = "332 - Von Gott will ich nicht lassen"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key a \minor
}


soprano = \relative c' {
  \partial 4 e4 
    a4 b4 c4 d4 
  b4. a8 g4 g4 
  a4 a4 b4 b4 
  e,2. e4 
  a4 b4 c4 d8 c8   %5
  b4. a8 g4 g4 
  a4 a4 b4 b4 
  e,2. e'4 
  d4 c4 c4 b4 
  c2. b4   %10
  c4 d4 e4 e4 
  d4. c8 b4 b4 
  c4 b4 a4 a4 
  b4. a8 g4 g8 f8 
  e4 a4 a8 b8 gis4   %15
  a2. 
  
}

alto = \relative c' {
  \partial 4 c8  d8 
    e4 e4 e4 d4 
  d8 e8 fis4 e4 e4 
  e4 d4 d8 f8 e8 d8 
  c2. b4 
  e4 fis8 gis8 a4 a8 g8   %5
  f8 e8 d4 e4 d4 
  e4 e4 d8 e8 fis4 
  b,2. c4 
  a'4 a4 g4 g4 
  g2. g8 f8   %10
  e4 d4 c4 e4 
  a8 g8 a8 fis8 g4 g4 
  g8 f8 g8 e8 c4 c4 
  b8 cis8 dis4 e4 d4 
  d8 c8 c8 d8 e8 f8 e4   %15
  e2. 
  
}

tenor = \relative c' {
  \partial 4 a8  b8 
    c4 b4 a4 a4 
  b2 b4 c4 
  c8 b8 a4 g8 a8 b4 
  a2. gis4 
  a4 d4 e4 a,4~   %5
  a8 gis8 a4 b4 b4 
  a8 b8 a8 g8 fis8 g8 a4 
  gis2. a4 
  a4 d4 d8 e8 f8 d8 
  e2. d4   %10
  c4 g'4 g4 c,8 b8 
  a4 d4 d4 d4 
  g,4 c4 c8 b8 a8 g8 
  fis8 g8 a4 b4 g4 
  g4 a8 b8 c8 d8 b4   %15
  cis2. 
  
}

baixo = \relative c {
  \partial 4 a4 
    a'4 gis4 a8 g8 fis4 
  g4 dis4 e4 c4 
  f4 fis4 g4 gis4 
  a2. e8 d8 
  c4 b4 a8 g'8 f8 e8   %5
  d8 e8 f4 e4 b4 
  c4 cis4 d4 dis4 
  e2. a8 g8 
  f4 fis4 g4 g,4 
  c2. g'4   %10
  a4 b4 c8 b8 a8 g8 
  fis8 e8 fis8 d8 g4 g8 f8 
  e8 d8 e8 c8 f8 g8 f8 e8 
  dis8 e8 fis4 e4 b4 
  c4 f4 e8 d8 e8 e,8   %15
  a2. 
  
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