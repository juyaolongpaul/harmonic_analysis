\version "2.10.33"

\header {
  title = "224 - Das walt' Gott Vater und Gott Sohn"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key f \major
}


soprano = \relative c' {
  \partial 4 f4 
  a4 c4 c4 a4 
  f4 g4 a4 f4 
  c'4 c4 d4 e4 
  f8 e8 d4 c4 c4 
  a4 c4 d4 c4   %5
  bes4 a4 g4 c4 
  a8 bes8 c4 bes4 a8 g8 
  f4 g f
  
}

alto = \relative c' {
  \partial 4 c4 
  f4 f4 e4 e4~ 
  e8 d8 c4 c4 c4 
  f4 g4 a8 b8 c4 
  c4. b8 g4 g4~ 
  g8 f16 e16 f4 f4 f4   %5
  g8 c,8 c8 f4 e8 g4 
  f4 f4 f4 e4 
  d16 e16 f4 e8 c4
  
}

tenor = \relative c' {
  \partial 4 a4 
  c4 c4 g4 a4 
  a4 g4 f4 a4 
  a4 c4 f4 e4 
  a4 d,8 g16 f16 e4 c4 
  c4 c4. bes4 a8   %5
  g4 a8. bes16 c4 c4 
  c4 c4 d4 c8 bes8 
  a8 g16 f16 c'8. bes16 a4
  
}

baixo = \relative c {
  \partial 4 f4 
  f4 a,8 bes8 c4 cis4 
  d4 e4 f4 f4 
  f4. e8 d4 a'8 g8 
  f4 g4 c,4 e4 
  f4 a,8 f8 bes4 f'4~   %5
  f8 e8 f4 c4 e4 
  f4 a,4 bes4 c4 
  d4 c f
  
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