\version "2.10.33"

\header {
  title = "307 - Christus, der uns selig macht"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key a \minor
}


soprano = \relative c'' {
  e4 e4 e4 e4 
  d4 c4 b2 
  a4 gis4 a4 b8 c8 
  d4 c4 b2 
  a4 b4 c8 b8 a4   %5
  a8 g8 f4 e2 
  d4 cis4 d4 e8 f8 
  g4 f4 e2 
  a4 a4 a4 g4 
  f4 e4 d2   %10
  g4 a4 b4 c4 
  d4 e4 d4 c4 
  b2 a4 b4 
  c8 b8 a4 a8 g8 f4 
  e2 d4 cis4   %15
  d4 e8 f8 g4 a4 
  g4 f4 e2 
  
}

alto = \relative c'' {
  gis8 a8 b8 gis8 a4 a4 
  a8 e8 e4 e2 
  c8 d8 e4. fis8 gis8 a8 
  b4. a8 a4 gis4 
  a8 g8 f4 g4. f8   %5
  e4. d8 cis2 
  d8 g,8 a4 a8 b8 cis8 d8 
  e4. d8 cis2 
  c4 f4. e16 d16 e4~ 
  e8 d4 cis8 d2   %10
  b8 e4 d8 d4 e4 
  f8 g4 a4 g8 g4 
  g2 e8 fis8 g4 
  g4 a8 g8 f8 e8 d4 
  d4 cis4 d8 a8 bes8 g8   %15
  a8 b8 c8 d8 e4 f4~ 
  f8 e4 d16 c16 b2 
  
}

tenor = \relative c' {
  b4 e8 d8 c8 d8 e8 c8~ 
  c8 b4 a8 a4 gis4 
  a4 b4 c4 d8 e8 
  f8 e8 e4 e2 
  c4 d4 c4. d8   %5
  a4 a4 a2 
  f8 d8 e4 f4 g8 a8 
  bes8 a8 a4 a2 
  a8 bes8 c4 c4. bes8 
  a4. a16 g16 f2   %10
  g8 c4 d16 c16 b8 d8 g,8 c8~ 
  c8 b8 c8 e8 f4. e8 
  d2 c4 d4 
  c4. cis8 d4 a4 
  a2 a4 g8 e8   %15
  f4 g4 c4 c4~ 
  c8. bes16 a8 b16 a16 gis2 
  
}

baixo = \relative c {
  e8 fis8 gis8 e8 a8 b8 c8 a8 
  fis8 gis8 a8 a,8 e'2 
  f4 e8 d8 c4 b8 a8~ 
  a8 gis8 a8 c8 e2 
  f8 e8 d4 e4 f4   %5
  cis4 d4 a2 
  bes4 a8 g8 f4 e8 d8~ 
  d8 cis8 e8 f8 a2 
  f8 g8 a8 bes8 c4. cis8 
  d4 a4 d,2   %10
  e'4 fis4 g8 f8 e4 
  d8 g8 c,4. b8 c8 e8 
  g2 a4 g8 f8 
  e4 f8 e8 d8 e8 f8 g8 
  a2 f4 e4   %15
  d4 c4. bes8 a8 f8 
  c'4 d4 e2 
  
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