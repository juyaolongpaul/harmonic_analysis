\version "2.10.33"

\header {
  title = "267 - Vater unser im Himmelreich"
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
  f4 g4 a4 a4 
  c4 d4 f4 e4   %5
  d4 cis4 d4 d4 
  e4 d4 c4 b4 
  a4 b4 a4 d4 
  c4 b4 c8 bes8 a4 
  a4 g4 f4 a4   %10
  bes4 a4 a4 g4 
  f4 e4 d2 
  
}

alto = \relative c' {
  \partial 4 f4 
    e4 d4 d8 cis8 d4 
  d4 cis4 a4 e'4 
  d4. e8 f4 cis4 
  d4 d4 cis4 e4 
  f4 a4 g4 a8 g8   %5
  f4 e4 fis4 g4 
  g8 a8 b4 a4 gis4 
  a4. gis8 e4 e4 
  e8 f8 g4 g4 f4 
  f4. e8 f4 fis4   %10
  g8 f8 e4 d8 cis8 d8 e8 
  a,8 d8 d8 cis8 a2 
  
}

tenor = \relative c' {
  \partial 4 d4 
    a4 a4 g4 f8 g8 
  a4 a8 g8 f4 a4 
  bes8 c8 d4 c8 d8 e4 
  a,4 bes4 e,4 c'8 bes8 
  a4 f'8 e8 d4 e4   %5
  a,4 a4 a4 b4 
  c4 f4 e4 d4 
  c4 f8 e8 cis4 b4 
  c4 d4 c4 c4 
  d4. c8 aes4 ees'4   %10
  d4. cis8 d2 
  d8 a8 bes8 a8 fis2 
  
}

baixo = \relative c {
  \partial 4 d4 
    cis4 d4 e4 d8 e8 
  f8 g8 a8 a,8 d4 c4 
  bes4 bes'4 a8 g8 f8 e8 
  d8 c8 bes4 a4 a'8 g8 
  f8 e8 d8 c8 b4 cis4   %5
  d4 a4 d4 g4 
  c8 b8 a8 gis8 a8 g8 f8 e8 
  f8 e8 d8 e8 a,4 gis4 
  a8 a'8 g8 f8 e4 f8 e8 
  d8 c8 bes8 c8 des4 c4   %10
  g4 a4 bes8 a8 b8 cis8 
  d8 f8 g8 a8 d,2 
  
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