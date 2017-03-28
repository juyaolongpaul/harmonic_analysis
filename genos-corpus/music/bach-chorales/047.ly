\version "2.10.33"

\header {
  title = "47 - Vater unser im Himmelreich"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key d \minor
}


soprano = \relative c'' {
  \partial 4 a4 
    a4 f4 g4 a4 
  f4 e4 d4 a'4 
  a4 g4 c4 a4 
  f4 g4 a4 a8 b8 
  c4 d8 e8 f4 e4   %5
  d4 cis4 d4 d4 
  e4 d4 c4 b4 
  a4 gis4 a4 d4 
  c4 b4 c4 a4 
  a4 g4 f4 a4   %10
  bes4 a8 g8 f4 g4 
  f4 e4 d
  
}

alto = \relative c' {
  \partial 4 f4 
    e4 d4 cis4 d8 e8 
  f4 cis4 a4 f'4 
  e4 d4 e4 f4 
  f4 e8 d8 e4 e4 
  a4 g4 f4 g4   %5
  f8 e8 e4 f4 f4 
  e4 b'4 e,4 e8 d8 
  c8 d8 e4 e4 g4 
  g4 f4 g4 f4 
  f4 e4 c4 d4   %10
  d4 e4 d4 d4 
  d4 cis4 a
  
}

tenor = \relative c' {
  \partial 4 d4 
    a4 a4 g4 d'4 
  d4 a4 f4 d'4 
  a4 bes4 c4 c4 
  d8 cis8 d4 cis4 c4 
  c8 a8 b8 cis8 d8 c8 bes4   %5
  b4 a4 a4 b4~ 
  b8 a4 gis8 a4 gis4 
  a4 b4 c4 b4 
  c4 d4 e4 d4 
  d8 c8 bes4 a4 a4   %10
  g8 f8 e4 a4 g8 a8 
  b4 e,8 f16 g16 fis4
  
}

baixo = \relative c {
  \partial 4 d4 
    cis4 d4 e4 f8 g8 
  a4 a,4 d4 d4 
  c4 bes4 a8 c8 f4 
  bes4 bes4 a4 a8 g8 
  f4. e8 d4 g4   %5
  gis4 a4 d,4 g4 
  c,4 b4 a4 e'4 
  f4 e4 a,4 g'8 f8 
  e4 d4 c4 d8 c8 
  bes4 c4 f,4 fis'4   %10
  g4 cis,4 d8 c8 bes8 a8 
  gis4 a4 d
  
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